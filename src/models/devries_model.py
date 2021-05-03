import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from src.models.networks import get_network


class net(pl.LightningModule):

    def __init__(self, cf):
        super(net, self).__init__()

        self.save_hyperparameters()

        self.optimizer_cfgs = cf.trainer.optimizer
        self.lr_scheduler_cfgs = cf.trainer.lr_scheduler

        if not cf.trainer.no_val_mode:
            self.selection_metrics = cf.trainer.callbacks.model_checkpoint.selection_metric
            self.selection_modes = cf.trainer.callbacks.model_checkpoint.mode

        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.num_classes = cf.data.num_classes
        self.loss_criterion = nn.CrossEntropyLoss()
        self.lmbda = 0.1
        self.budget = cf.model.budget
        self.test_conf_scaling = cf.eval.test_conf_scaling
        self.ext_confid_name = cf.eval.ext_confid_name
        self.model = get_network(cf.model.network.name)(cf) # todo make explciit arguments in factory!!

    def forward(self, x):
        return self.model(x)

    def mcd_eval_forward(self, x, n_samples):
        # self.model.encoder.eval_mcdropout = True
        softmax_list = []
        conf_list =  []
        for _ in range(n_samples - len(softmax_list)):
            logits, confidence = self.model(x)
            softmax = F.softmax(logits, dim=1)
            confidence = torch.sigmoid(confidence).squeeze(1)
            softmax_list.append(softmax.unsqueeze(2))
            conf_list.append(confidence.unsqueeze(1))
        return torch.cat(softmax_list, dim=2), torch.cat(conf_list, dim=1)


    def training_step(self, batch, batch_idx):

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

            xentropy_loss = self.loss_criterion(pred_new, y)
            confidence_loss = torch.mean(-torch.log(confidence))

            loss = xentropy_loss + (self.lmbda * confidence_loss)
            # print(self.lmbda, confidence_loss.item())
            if self.budget > confidence_loss:
                self.lmbda = self.lmbda / 1.01
            elif self.budget <= confidence_loss:
                self.lmbda = self.lmbda / 0.99

        elif self.ext_confid_name == "tcp":
            logits, confidence = self.model(x)
            confidence = torch.sigmoid(confidence)
            pred_original = F.softmax(logits, dim=1)
            tcp = pred_original.gather(1, y.unsqueeze(1))
            loss =self.loss_criterion(logits, y) + 1e-2 * F.mse_loss(confidence, tcp)

        else:
            raise NotImplementedError

        return {"loss":loss, "softmax": pred_original, "labels": y, "confid": confidence.squeeze(1)}


    def validation_step(self, batch, batch_idx):

        pass


    def test_step(self, batch, batch_idx, *args):
        x, y = batch

        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax, confidence = self.mcd_eval_forward(x=x,
                                                 n_samples=self.test_mcd_samples,
                                                 )
        else:
            logits, confidence = self.model(x)
            softmax = F.softmax(logits, dim=1)
            confidence = torch.sigmoid(confidence).squeeze(1)

        self.test_results = {"softmax": softmax, "labels": y, "confid": confidence}


    def configure_optimizers(self):
        optimizers = [torch.optim.SGD(self.parameters(),
                               lr=self.optimizer_cfgs.learning_rate,
                               momentum=self.optimizer_cfgs.momentum,
                               nesterov = self.optimizer_cfgs.nesterov,
                               weight_decay=self.optimizer_cfgs.weight_decay)]

        schedulers = []
        if self.lr_scheduler_cfgs.name == "MultiStep":
            schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[0],
                                                               milestones=self.lr_scheduler_cfgs.milestones,
                                                               gamma=0.2,
                                                               verbose=True)]
        elif self.lr_scheduler_cfgs.name == "CosineAnnealing":
            schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0],
                                                                     T_max=self.lr_scheduler_cfgs.max_epochs,
                                                                     verbose=True)]

        return optimizers, schedulers


    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint at epoch {}".format(self.loaded_epoch))




