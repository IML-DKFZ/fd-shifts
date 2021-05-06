import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from src.models.networks import get_network


class net(pl.LightningModule):

    def __init__(self, cf):
        super(net, self).__init__()

        self.save_hyperparameters()

        self.test_mcd_samples = cf.model.test_mcd_samples
        self.monitor_mcd_samples = cf.model.monitor_mcd_samples
        self.num_epochs = cf.trainer.num_epochs

        self.loss_criterion = nn.CrossEntropyLoss()
        self.ext_confid_name = cf.eval.get("ext_confid_name")


        self.optimizer_cfgs = cf.trainer.optimizer
        self.lr_scheduler_cfgs = cf.trainer.lr_scheduler

        if not cf.trainer.no_val_mode:
            self.selection_metrics = cf.trainer.callbacks.model_checkpoint.selection_metric
            self.selection_modes = cf.trainer.callbacks.model_checkpoint.mode

        self.query_confids = cf.eval.confidence_measures
        self.num_classes = cf.data.num_classes

        self.model = get_network(cf.model.network.name)(cf) # todo make explciit arguemnts in factory!!

    def forward(self, x):
        return self.model(x)



    def mcd_eval_forward(self, x, n_samples):
        # self.model.encoder.eval_mcdropout = True
        self.model.encoder.enable_dropout()

        softmax_list = []
        for _ in range(n_samples - len(softmax_list)):
            logits = self.model(x)
            softmax = F.softmax(logits, dim=1)
            softmax_list.append(softmax.unsqueeze(2))

        self.model.encoder.disable_dropout()

        return torch.cat(softmax_list, dim=2)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)
        return {"loss":loss, "softmax": softmax, "labels": y}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        softmax_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["val"]):
            softmax_dist = self.mcd_eval_forward(x=x,  n_samples=self.monitor_mcd_samples)

        if self.current_epoch == self.num_epochs - 1:
            # save mcd output for psuedo-test if actual test is with mcd.
            if any("mcd" in cfd for cfd in self.query_confids["test"]) and softmax_dist is None:
                    softmax_dist = self.mcd_eval_forward(x=x,
                                                         n_samples=self.monitor_mcd_samples)

        return {"loss":loss, "softmax": softmax, "labels": y, "softmax_dist": softmax_dist}


    def test_step(self, batch, batch_idx, *args):
        x, y = batch
        logits = self.model(x)
        softmax = F.softmax(logits, dim=1)

        softmax_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax_dist = self.mcd_eval_forward(x=x, n_samples=self.test_mcd_samples)

        self.test_results = {"softmax": softmax, "labels": y, "softmax_dist": softmax_dist}


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




