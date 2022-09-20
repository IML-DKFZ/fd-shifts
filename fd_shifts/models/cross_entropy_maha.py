import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from fd_shifts.models.networks import get_network
from tqdm import tqdm
import pl_bolts
from fd_shifts.utils.exp_utils import GradualWarmupSchedulerV2
import numpy as np


class net(pl.LightningModule):
    def __init__(self, cf):
        super(net, self).__init__()

        self.save_hyperparameters()
        self.optimizer_cfgs = cf.trainer.optimizer
        self.lr_scheduler_cfgs = cf.trainer.lr_scheduler
        self.trainer_cfgs = cf.trainer
        self.test_mcd_samples = cf.model.test_mcd_samples
        self.monitor_mcd_samples = cf.model.monitor_mcd_samples
        self.learning_rate = cf.trainer.learning_rate

        self.lr_scheduler = cf.trainer.lr_scheduler
        self.momentum = cf.trainer.momentum
        self.weight_decay = cf.trainer.weight_decay
        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction="sum")
        self.ext_confid_name = dict(cf.eval).get("ext_confid_name")
        self.latent = []
        self.labels = []
        self.network = get_network(cf.model.network.name)(cf)

        self.mean = torch.zeros(
            (self.hparams.cf.data.num_classes, self.hparams.cf.model.fc_dim)
        )
        self.icov = torch.eye(self.hparams.cf.model.fc_dim)

    def forward(self, x):
        return self.network(x)

    def mcd_eval_forward(self, x, n_samples):
        # self.model.encoder.eval_mcdropout = True
        self.enable_dropout()

        softmax_list = []
        conf_list = []
        for _ in range(n_samples - len(softmax_list)):
            z = self.model.forward_features(x)
            probs = self.model.head(z)
            softmax = torch.softmax(probs, dim=1)
            zm = z[:, None, :] - self.mean

            maha = -(torch.einsum("inj,jk,ink->in", zm, self.icov, zm))
            maha = maha.max(dim=1)[0].type_as(x)

            softmax_list.append(softmax.unsqueeze(2))
            conf_list.append(maha.unsqueeze(1))

        self.disable_dropout()

        return torch.cat(softmax_list, dim=2), torch.cat(conf_list, dim=1)

    def on_train_start(self):
        for ix, x in enumerate(self.network.named_modules()):
            tqdm.write(str(ix))
            tqdm.write(str(x[1]))

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.network.forward_features(x)
        self.latent.append(z.cpu())
        self.labels.append(y.cpu())

        logits = self.network(x)
        loss = self.loss_ce(logits, y)
        softmax = F.softmax(logits, dim=1)
        return {"loss": loss, "softmax": softmax, "labels": y, "confid": None}

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            z = torch.cat(self.latent, dim=0)
            y = torch.cat(self.labels, dim=0)

            mean = []
            for c in y.unique():
                mean.append(z[y == c].mean(dim=0))

            mean = torch.stack(mean, dim=0)
            self.mean = mean
            self.icov = torch.inverse(
                torch.tensor(np.cov(z.numpy(), rowvar=False)).type_as(
                    self.network.encoder.model.classifier.weight
                )
            ).cpu()

        self.latent = []
        self.labels = []

    def training_step_end(self, batch_parts):
        batch_parts["loss"] = batch_parts["loss"].mean()
        return batch_parts

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.network(x)
        loss = self.loss_ce(logits, y)
        softmax = F.softmax(logits, dim=1)
        return {"loss": loss, "softmax": softmax, "labels": y, "confid": None}

    def validation_step_end(self, batch_parts):
        return batch_parts

    def on_test_start(self, *args):
        tqdm.write("Calculating trainset mean and cov")
        all_z = []
        all_y = []
        for x, y in tqdm(self.trainer.datamodule.train_dataloader()):
            x = x.type_as(self.network.encoder.model.classifier.weight)
            y = y.type_as(self.network.encoder.model.classifier.weight)
            z = self.network.forward_features(x)
            all_z.append(z.cpu())
            all_y.append(y.cpu())
            break

        all_z = torch.cat(all_z, dim=0)
        all_y = torch.cat(all_y, dim=0)

        mean = []
        for c in all_y.unique():
            mean.append(all_z[all_y == c].mean(dim=0))

        mean = torch.stack(mean, dim=0)
        self.mean = mean.type_as(self.network.encoder.model.classifier.weight)
        self.icov = torch.inverse(
            torch.tensor(np.cov(all_z.numpy(), rowvar=False)).type_as(
                self.network.encoder.model.classifier.weight
            )
        )

    def test_step(self, batch, batch_idx, *args):
        x, y = batch

        softmax = F.softmax(self.network(x), dim=1)
        z = self.network.forward_features(x)
        maha = None
        if any("ext" in cfd for cfd in self.query_confids["test"]):
            zm = z[:, None, :] - self.mean

            maha = -(torch.einsum("inj,jk,ink->in", zm, self.icov, zm))
            maha = maha.max(dim=1)[0].type_as(x)
            # maha final ist abstand zu most likely class

        softmax_dist = None
        confid_dist = None

        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax_dist, confid_dist = self.mcd_eval_forward(
                x=x, n_samples=self.test_mcd_samples
            )

        self.test_results = {
            "softmax": softmax,
            "softmax_dist": softmax_dist,
            "labels": y,
            "confid": maha,
            "confid_dist": confid_dist,
            "encoded": z,
        }

    def configure_optimizers(self):
        if self.optimizer_cfgs.name == "SGD":
            optimizers = [
                torch.optim.SGD(
                    self.network.parameters(),
                    lr=self.optimizer_cfgs.learning_rate,
                    momentum=self.optimizer_cfgs.momentum,
                    nesterov=self.optimizer_cfgs.nesterov,
                    weight_decay=self.optimizer_cfgs.weight_decay,
                )
            ]
        if self.optimizer_cfgs.name == "ADAM":
            optimizers = [
                torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.optimizer_cfgs.learning_rate,
                    # momentum=self.hparams.trainer.momentum,
                    weight_decay=self.optimizer_cfgs.weight_decay,
                )
            ]
        schedulers = []
        if self.lr_scheduler_cfgs.name == "MultiStep":
            schedulers = [
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizers[0],
                    milestones=self.lr_scheduler_cfgs.milestones,
                    gamma=self.lr_scheduler_cfgs.gamma,
                    verbose=True,
                )
            ]
        elif self.lr_scheduler_cfgs.name == "CosineAnnealing":
            schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], T_max=self.lr_scheduler_cfgs.max_epochs, verbose=True
                )
            ]
        elif self.lr_scheduler_cfgs.name == "CosineAnnealingWarmRestarts":
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizers[0], (self.lr_scheduler_cfgs.max_epochs - 1) * 4
            )
            scheduler_warmup = GradualWarmupSchedulerV2(
                optimizers[0],
                multiplier=10,
                total_epoch=1,
                after_scheduler=scheduler_cosine,
            )  # try basic cosine annealing

            # lr_sched = {
            #    "scheduler": scheduler_cosine,
            #    "interval": "step",
            # }

            lr_sched = [
                {"scheduler": scheduler_warmup, "interval": "epoch", "frequency": 1}
            ]

        elif self.lr_scheduler_cfgs.name == "LinearWarmupCosineAnnealing":
            num_batches = (
                len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
            )
            schedulers = [
                {
                    "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                        optimizer=optimizers[0],
                        max_epochs=self.lr_scheduler_cfgs.max_epochs * num_batches,
                        warmup_epochs=self.lr_scheduler_cfgs.warmup_epochs,
                    ),
                    "interval": "step",
                }
            ]

        return optimizers, schedulers

    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint at epoch {}".format(self.loaded_epoch))

    def load_only_state_dict(self, path):
        ckpt = torch.load(path)
        print("loading checkpoint from epoch {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)
