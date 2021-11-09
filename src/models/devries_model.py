import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import pl_bolts
from src.models.networks import get_network
from tqdm import tqdm

class net(pl.LightningModule):

    def __init__(self, cf):
        super(net, self).__init__()

        self.save_hyperparameters()

        self.optimizer_cfgs = cf.trainer.optimizer
        self.lr_scheduler_cfgs = cf.trainer.lr_scheduler

        if cf.trainer.callbacks.model_checkpoint is not None:
            print("Initializing custom Model Selector.", cf.trainer.callbacks.model_checkpoint)
            self.selection_metrics = cf.trainer.callbacks.model_checkpoint.selection_metric
            self.selection_modes = cf.trainer.callbacks.model_checkpoint.mode

        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.num_classes = cf.data.num_classes
        self.nll_loss = nn.NLLLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lmbda = 0.1
        self.budget = cf.model.budget
        self.test_conf_scaling = cf.eval.test_conf_scaling
        self.ext_confid_name = cf.eval.ext_confid_name
        self.imagenet_weights_path = dict(cf.model.network).get("imagenet_weights_path")

        if self.ext_confid_name == "dg":
            self.reward = cf.model.dg_reward
            self.pretrain_epochs = cf.trainer.dg_pretrain_epochs
            self.load_dg_backbone_path = dict(cf.model.network).get("load_dg_backbone_path")
            self.save_dg_backbone_path = dict(cf.model.network).get("save_dg_backbone_path")

        self.model = get_network(cf.model.network.name)(cf) # todo make explciit arguments in factory!!

        self.test_mcd_samples = cf.model.test_mcd_samples
        self.monitor_mcd_samples = cf.model.monitor_mcd_samples


    def forward(self, x):
        return self.model(x)


    def mcd_eval_forward(self, x, n_samples):
        # self.model.encoder.eval_mcdropout = True
        self.model.encoder.enable_dropout()

        softmax_list = []
        conf_list =  []
        for _ in range(n_samples - len(softmax_list)):
            if self.ext_confid_name == "devries":
                logits, confidence = self.model(x)
                softmax = F.softmax(logits, dim=1)
                confidence = torch.sigmoid(confidence).squeeze(1)
                softmax_list.append(softmax.unsqueeze(2))
                conf_list.append(confidence.unsqueeze(1))
            if self.ext_confid_name == "dg":
                outputs = self.model(x)
                outputs = F.softmax(outputs, dim=1)
                softmax, reservation = outputs[:, :-1], outputs[:, -1]
                confidence = 1 - reservation
                softmax_list.append(softmax.unsqueeze(2))
                conf_list.append(confidence.unsqueeze(1))

        self.model.encoder.disable_dropout()

        return torch.cat(softmax_list, dim=2), torch.cat(conf_list, dim=1)






    # def on_train_epoch_start(self):

        # if self.current_epoch == self.pretrain_epochs and self.ext_confid_name == "dg" and self.load_dg_backbone_path is not None:
        #
        #     loaded_ckpt = torch.load(self.load_dg_backbone_path)
        #     loaded_state_dict = loaded_ckpt["state_dict"]
        #     # self.load_state_dict(loaded_state_dict, strict=True)
        #     self.load_from_checkpoint(self.load_dg_backbone_path)
        #     print("loaded pretrained dg backbone from {}".format(self.load_dg_backbone_path))

    def on_epoch_end(self):

        if self.ext_confid_name == "dg" and self.current_epoch == self.pretrain_epochs -1 and self.save_dg_backbone_path is not None:
            self.trainer.save_checkpoint(self.save_dg_backbone_path)
            tqdm.write("saved pretrained dg backbone to {}".format(self.save_dg_backbone_path))

    def on_train_start(self):

        if self.imagenet_weights_path:
            self.model.encoder.load_pretrained_imagenet_params(self.imagenet_weights_path)

        if self.current_epoch > 0: # check if resumed training
            tqdm.write("stepping scheduler after resume...")
            self.trainer.lr_schedulers[0]["scheduler"].step()

        for ix, x in enumerate(self.model.named_modules()):
            tqdm.write(str(ix))
            tqdm.write(str(x[1]))
            if isinstance(x[1], nn.Conv2d) or isinstance(x[1], nn.Linear):
                tqdm.write(str(x[1].weight.mean().item()))




    def training_step(self, batch, batch_idx):

        x, y = batch
        if self.ext_confid_name == "devries":
            logits, confidence = self.model(x)
            confidence = torch.sigmoid(confidence)
            pred_original = F.softmax(logits, dim=1)
            labels_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
            # print(x.mean().item(), logits.mean().item())
            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).cuda()
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (
                        1 - conf.expand_as(labels_onehot))
            pred_new = torch.log(pred_new)

            xentropy_loss = self.nll_loss(pred_new, y)
            confidence_loss = torch.mean(-torch.log(confidence))

            loss = xentropy_loss + (self.lmbda * confidence_loss)
            # print(self.lmbda, confidence_loss.item())
            if self.budget > confidence_loss:
                self.lmbda = self.lmbda / 1.01
            elif self.budget <= confidence_loss:
                self.lmbda = self.lmbda / 0.99

        elif self.ext_confid_name == "dg":
            logits = self.model(x)
            softmax = F.softmax(logits, dim=1)
            pred_original, reservation = softmax[:, :-1], softmax[:, -1]
            confidence = 1 - reservation.unsqueeze(1)
            # print("CHECK CONF", confidence.mean().item(), confidence.std().item(), confidence.min().item(), confidence.max().item())
            if self.current_epoch >= self.pretrain_epochs and self.reward > -1:
                gain = torch.gather(pred_original, dim=1, index=y.unsqueeze(1)).squeeze()
                doubling_rate = (gain.add(reservation.div(self.reward))).log()
                loss = -doubling_rate.mean().unsqueeze(0)
                # print(x.mean().item(), logits.mean().item(), logits.min().item(), logits.max().item(),
                #       loss.mean().item(), "CHECK")
            else:
                loss = self.cross_entropy_loss(logits[:, :-1], y)
                # print(x.mean().item(), logits.mean().item(), logits.min().item(), logits.max().item(),
                #       loss.mean().item(), "CHECK")

        return {"loss":loss, "softmax": pred_original, "labels": y, "confid": confidence.squeeze(1)} # ,"imgs":x

    def training_step_end(self, batch_parts):
        batch_parts["loss"] = batch_parts["loss"].mean()
        return batch_parts

    def validation_step(self, batch, batch_idx):

        x, y = batch

        if self.ext_confid_name == "devries":
            logits, confidence = self.model(x)
            confidence = torch.sigmoid(confidence)
            pred_original = F.softmax(logits, dim=1)
            labels_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)

            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).cuda()
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (
                    1 - conf.expand_as(labels_onehot))
            pred_new = torch.log(pred_new)

            xentropy_loss = self.nll_loss(pred_new, y)
            confidence_loss = torch.mean(-torch.log(confidence))

            loss = xentropy_loss + (self.lmbda * confidence_loss)

        elif self.ext_confid_name == "dg":
            outputs = self.model(x)
            outputs = F.softmax(outputs, dim=1)
            pred_original, reservation = outputs[:, :-1], outputs[:, -1]
            confidence = 1 - reservation.unsqueeze(1)
            if self.current_epoch >= self.pretrain_epochs and self.reward > -1:
                gain = torch.gather(pred_original, dim=1, index=y.unsqueeze(1)).squeeze()
                doubling_rate = (gain.add(reservation.div(self.reward))).log()
                loss = -doubling_rate.mean()
            else:
                loss = self.cross_entropy_loss(outputs[:, :-1], y)

        # print(self.lmbda, confidence_loss.item())
        # print(x.mean(), pred_original.std())
        return {"loss": loss, "softmax": pred_original, "labels": y, "confid": confidence.squeeze(1)}

    def validation_step_end(self, batch_parts):
        return batch_parts

    def test_step(self, batch, batch_idx, *args):
        x, y = batch
        if self.ext_confid_name == "devries":
            logits, confidence = self.model(x)
            softmax = F.softmax(logits, dim=1)
            confidence = torch.sigmoid(confidence).squeeze(1)
        elif self.ext_confid_name == "dg":
            outputs = self.model(x)
            outputs = F.softmax(outputs, dim=1)
            softmax, reservation = outputs[:, :-1], outputs[:, -1]
            confidence = 1 - reservation

        softmax_dist = None
        confid_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax_dist, confid_dist = self.mcd_eval_forward(x=x, n_samples=self.test_mcd_samples)
            # print(softmax_dist.std(1).mean(), confid_dist.std(1).mean(), confid_dist[0])

        self.test_results = {"softmax": softmax, "labels": y, "confid": confidence, "softmax_dist": softmax_dist, "confid_dist": confid_dist}


    def configure_optimizers(self):
        optimizers = [torch.optim.SGD(self.model.parameters(),
                               lr=self.optimizer_cfgs.learning_rate,
                               momentum=self.optimizer_cfgs.momentum,
                               nesterov = self.optimizer_cfgs.nesterov,
                               weight_decay=self.optimizer_cfgs.weight_decay)]

        schedulers = []
        if self.lr_scheduler_cfgs.name == "MultiStep":
            schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[0],
                                                               milestones=self.lr_scheduler_cfgs.milestones,
                                                               gamma=self.lr_scheduler_cfgs.gamma,
                                                               verbose=True)]
        elif self.lr_scheduler_cfgs.name == "CosineAnnealing":
            schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0],
                                                                     T_max=self.lr_scheduler_cfgs.max_epochs,
                                                                     verbose=True)]
        elif self.lr_scheduler_cfgs.name == "LinearWarmupCosineAnnealing":
            num_batches = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
            schedulers = [{
                "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                    optimizer=optimizers[0],
                    max_epochs=self.lr_scheduler_cfgs.max_epochs * num_batches,
                    warmup_epochs=self.lr_scheduler_cfgs.warmup_epochs,
                ),
                "interval": "step",
            }]


        return optimizers, schedulers


    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint from epoch {}".format(self.loaded_epoch))

    def load_only_state_dict(self, path):
        ckpt = torch.load(path)
        print("loading checkpoint from epoch {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)




