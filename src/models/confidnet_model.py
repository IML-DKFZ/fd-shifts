import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from src.utils import exp_utils
from src.models.networks import get_network


class net(pl.LightningModule):

    def __init__(self, cf):
        super(net, self).__init__()

        self.save_hyperparameters()

        self.test_mcd_samples = cf.model.test_mcd_samples
        self.monitor_mcd_samples = cf.model.monitor_mcd_samples
        self.learning_rate = cf.trainer.learning_rate
        self.learning_rate_confidnet = cf.trainer.learning_rate_confidnet
        self.learning_rate_confidnet_finetune = cf.trainer.learning_rate_confidnet_finetune
        self.momentum = cf.trainer.momentum
        self.weight_decay = cf.trainer.weight_decay
        self.global_seed = cf.trainer.global_seed
        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.selection_metrics = cf.callbacks.model_checkpoint.selection_metric
        self.selection_modes = cf.callbacks.model_checkpoint.mode

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction="sum")

        # self.backbone = get_network(cf.model.network.backbone)(cf) # todo make explciit arguemnts in factory!!
        self.network = get_network(cf.model.network.name)(cf) # todo make explciit arguemnts in factory!!
        self.training_stage = 0 # will be iincreased by TrainingStages callback

    def forward(self, x):
        return self.network(x)


    def on_train_start(self):
        exp_utils.set_seed(self.global_seed)


    def training_step(self, batch, batch_idx):
        if self.training_stage == 0 and 1==2:
            x, y = batch
            logits = self.backbone(x)
            loss = self.loss_ce(logits, y)
            softmax = F.softmax(logits, dim=1)
            confid = None
            return {"loss":loss, "softmax": softmax, "labels": y, "confid": confid}

        if self.training_stage > 0 or 1==1:
            x, y = batch
            pred_confid = self.network(x).squeeze()
            # logits = self.backbone(x)
            # pred_confid = logits
            # softmax = F.softmax(logits, dim=1)
            # tcp = softmax.gather(1, y.unsqueeze(1))
            # print(softmax.size(), y.size(), tcp.size(), pred_confid.size())
            # print(tcp.min(), tcp.max(), pred_confid.min(), pred_confid.max())
            loss = torch.mean((pred_confid))
            softmax = pred_confid
            print(loss)
            return {"loss":loss, "softmax": softmax, "labels": y, "confid": pred_confid}

    def on_after_backward(self):
        if self.trainer.global_step % 25000000 == 0 and self.training_stage>0:
            for k, v in self.named_parameters():
                try:
                    min= v.min()
                    max = v.max()
                except:
                    min= None
                    max = None
                print(k, min , max)
        # if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
        #     for k, v in self.named_parameters():
        #         self.logger[0].experiment.add_histogram(
        #             tag=k, values=v.grad, global_step=self.trainer.global_step
        #         )

    def validation_step(self, batch, batch_idx):

        if self.training_stage == 0 or 1==2:
            x, y = batch
            logits = self.network(x)
            # loss = self.loss_ce(logits, y)
            loss = self.loss_mse(logits, logits + 1e-3)
            softmax = F.softmax(logits, dim=1)
            confid = None
            return {"loss":loss, "softmax": softmax, "labels": y, "confid": confid}

        if self.training_stage > 0 and 1==2:
            x, y = batch
            pred_confid = self.network(x)
            logits = self.backbone(x)
            softmax = F.softmax(logits, dim=1)
            tcp = softmax.gather(1, y.unsqueeze(1))
            loss = self.loss_mse(pred_confid, tcp)
            return {"loss":loss, "softmax": softmax, "labels": y, "confid": pred_confid}



    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, confid = self.network(x)
        softmax = F.softmax(logits, dim=1)

        softmax_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax_dist = self.mcd_eval_forward(x=x,
                                            n_samples=self.test_mcd_samples,
                                            existing_softmax_list=[softmax.unsqueeze(2)])

        return {"softmax": softmax, "softmax_dist": softmax_dist, "labels": y}


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.learning_rate,
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)


    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint at epoch {}".format(self.loaded_epoch))




