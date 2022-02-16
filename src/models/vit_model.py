import pytorch_lightning as pl
import pl_bolts
import timm
import torch
import torch.nn as nn
import hydra
import numpy as np
from pytorch_lightning.utilities.parsing import AttributeDict
from tqdm import tqdm


class net(pl.LightningModule):
    def __init__(self, cf):
        super().__init__()

        self.hparams: AttributeDict = AttributeDict()

        self.save_hyperparameters()
        self.hparams.update(dict(cf))

        self.model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=True,
            img_size=self.hparams.data.img_size[0],
            num_classes=self.hparams.data.num_classes,
            drop_rate=self.hparams.model.dropout_rate * 0.1,
        )
        self.model.reset_classifier(self.hparams.data.num_classes)
        self.model.head.weight.tensor = torch.zeros_like(self.model.head.weight)
        self.model.head.bias.tensor = torch.zeros_like(self.model.head.bias)

        self.mean = torch.zeros(
            (self.hparams.data.num_classes, self.model.num_features)
        )
        self.icov = torch.eye(self.model.num_features)

        self.ext_confid_name = self.hparams.eval.ext_confid_name
        self.latent = []
        self.labels = []

        self.query_confids = cf.eval.confidence_measures
        self.test_mcd_samples = 50

    def disable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].eval()

    def enable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].train()

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model.forward_features(x)
        probs = self.model.head(z)
        loss = torch.nn.functional.cross_entropy(probs, y)

        self.latent.append(z.cpu())
        self.labels.append(y.cpu())

        return {"loss": loss, "softmax": torch.softmax(probs, dim=1), "labels": y}

    def training_step_end(self, batch_parts):
        batch_parts["loss"] = batch_parts["loss"].mean()
        return batch_parts

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
                    self.model.head.weight
                )
            ).cpu()

        self.latent = []
        self.labels = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if dataloader_idx > 0:
            y = y.fill_(0)
            y = y.long()

        z = self.model.forward_features(x)
        zm = z[:, None, :].cpu() - self.mean

        maha = -(torch.einsum("inj,jk,ink->in", zm, self.icov, zm))
        maha = maha.max(dim=1)[0]

        probs = self.model.head(z)
        loss = torch.nn.functional.cross_entropy(probs, y)

        return {
            "loss": loss,
            "softmax": torch.softmax(probs, dim=1),
            "labels": y,
            "confid": maha.type_as(x),
        }

    def validation_step_end(self, batch_parts):
        return batch_parts

    def on_test_start(self, *args):
        tqdm.write("Calculating trainset mean and cov")
        all_z = []
        all_y = []
        for x, y in tqdm(self.trainer.datamodule.train_dataloader()):
            x = x.type_as(self.model.head.weight)
            y = y.type_as(self.model.head.weight)
            z = self.model.forward_features(x)
            all_z.append(z.cpu())
            all_y.append(y.cpu())

        all_z = torch.cat(all_z, dim=0)
        all_y = torch.cat(all_y, dim=0)

        mean = []
        for c in all_y.unique():
            mean.append(all_z[all_y == c].mean(dim=0))

        mean = torch.stack(mean, dim=0)
        self.mean = mean.type_as(self.model.head.weight)
        self.icov = torch.inverse(
            torch.tensor(np.cov(all_z.numpy(), rowvar=False)).type_as(
                self.model.head.weight
            )
        )

    def test_step(self, batch, batch_idx, *args):
        x, y = batch
        z = self.model.forward_features(x)
        zm = z[:, None, :] - self.mean

        maha = -(torch.einsum("inj,jk,ink->in", zm, self.icov, zm))
        maha = maha.max(dim=1)[0]
        print(maha)

        probs = self.model.head(z)

        softmax_dist = None
        confid_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax_dist, confid_dist = self.mcd_eval_forward(
                x=x, n_samples=self.test_mcd_samples
            )

        self.test_results = {
            "softmax": torch.softmax(probs, dim=1),
            "labels": y,
            "confid": maha.type_as(x),
            "softmax_dist": softmax_dist,
            "confid_dist": confid_dist,
        }

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.trainer.learning_rate,
            momentum=self.hparams.trainer.momentum,
            weight_decay=self.hparams.trainer.weight_decay,
        )

        lr_sched = {
            "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer=optim,
                max_epochs=self.hparams.trainer.num_steps,
                warmup_start_lr=self.hparams.trainer.lr_scheduler.warmup_start_lr,
                warmup_epochs=self.hparams.trainer.lr_scheduler.warmup_epochs,
                eta_min=self.hparams.trainer.lr_scheduler.eta_min,
            ),
            "interval": "step",
        }

        optimizers = {
            "optimizer": optim,
            "lr_scheduler": lr_sched,
            "frequency": 1,
        }

        return optimizers

    def load_only_state_dict(self, path):
        ckpt = torch.load(path)
        print("loading checkpoint from epoch {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)
