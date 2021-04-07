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
        self.learning_rate = cf.trainer.learning_rate
        self.momentum = cf.trainer.momentum
        self.weight_decay = cf.trainer.weight_decay
        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.selection_metrics = cf.trainer.callbacks.model_checkpoint.selection_metric
        self.selection_modes = cf.trainer.callbacks.model_checkpoint.mode

        self.loss_criterion = nn.CrossEntropyLoss()

        self.encoder = get_network(cf.model.network.name)(cf) # todo make explciit arguemnts in factory!!

    def forward(self, x):
        return self.encoder(x)


    def mcd_eval_forward(self, x, n_samples, existing_softmax_list=None):
        self.encoder.eval_mcdropout = True
        softmax_list = existing_softmax_list if existing_softmax_list is not None else []
        for _ in range(n_samples - len(softmax_list)):
            softmax_list.append(F.softmax(self.encoder(x), dim=1).unsqueeze(2))
        self.encoder.eval_mcdropout = False

        return torch.cat(softmax_list, dim=2)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)
        return {"loss":loss, "softmax": softmax, "labels": y}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        softmax_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["val"]):
            softmax_dist = self.mcd_eval_forward(x=x,
                                                 n_samples=self.monitor_mcd_samples,
                                                 existing_softmax_list=[softmax.unsqueeze(2)])


        if self.current_epoch == self.num_epochs - 1:
            # save mcd output for psuedo-test if actual test is with mcd.
            if any("mcd" in cfd for cfd in self.query_confids["test"]):
                if softmax_dist is None:
                    softmax_dist = self.mcd_eval_forward(x=x,
                                                         n_samples=self.monitor_mcd_samples,
                                                         existing_softmax_list=[softmax.unsqueeze(2)])

        return {"loss":loss, "softmax": softmax, "labels": y, "softmax_dist": softmax_dist}


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        softmax = F.softmax(logits, dim=1)

        softmax_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax_dist = self.mcd_eval_forward(x=x,
                                            n_samples=self.test_mcd_samples,
                                            existing_softmax_list=[softmax.unsqueeze(2)])

        self.test_results = {"softmax": softmax, "softmax_dist": softmax_dist, "labels": y}


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.learning_rate,
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)


    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint at epoch {}".format(self.loaded_epoch))




