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
        self.learning_rate_confidnet = cf.trainer.learning_rate_confidnet
        self.learning_rate_confidnet_finetune = cf.trainer.learning_rate_confidnet_finetune
        self.momentum = cf.trainer.momentum
        self.weight_decay = cf.trainer.weight_decay
        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.selection_metrics = cf.callbacks.model_checkpoint.selection_metric
        self.selection_modes = cf.callbacks.model_checkpoint.mode
        self.pretrained_backbone_path = cf.callbacks.training_stages.pretrained_backbone_path
        self.pretrained_confidnet_path = cf.callbacks.training_stages.pretrained_confidnet_path

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction="sum")

        self.network = get_network(cf.model.network.name)(cf) # todo make explciit arguemnts in factory!!
        self.backbone = get_network(cf.model.network.backbone)(cf)  # todo make explciit arguemnts in factory!!
        self.training_stage = 0 # will be iincreased by TrainingStages callback

    def forward(self, x):
        return self.network(x)

    def mcd_eval_forward(self, x, n_samples, existing_softmax_list=None):
        self.backbone.encoder.eval_mcdropout = True
        softmax_list = existing_softmax_list if existing_softmax_list is not None else []
        for _ in range(n_samples - len(softmax_list)):
            softmax_list.append(F.softmax(self.backbone(x), dim=1).unsqueeze(2))
        self.backbone.encoder.eval_mcdropout = False
        return torch.cat(softmax_list, dim=2)

    def training_step(self, batch, batch_idx):
        if self.training_stage == 0:
            x, y = batch
            logits = self.backbone(x)
            loss = self.loss_ce(logits, y)
            softmax = F.softmax(logits, dim=1)
            return {"loss":loss, "softmax": softmax, "labels": y, "confid": None}

        if self.training_stage == 1:
            x, y = batch
            outputs = self.network(x)
            softmax = F.softmax(outputs[0], dim=1)
            pred_confid = torch.sigmoid(outputs[1])
            tcp = softmax.gather(1, y.unsqueeze(1))
            # print("CHECK PRED CONFID", pred_confid.mean(), pred_confid.min(), pred_confid.max())
            # print("CHECK TCP", tcp.mean(), tcp.min(), tcp.max())
            # print(pred_confid[0].item(), y[0].item(), tcp[0].item(), softmax[0])
            loss = F.mse_loss(pred_confid, tcp) # self.loss_mse(pred_confid, tcp) #
            return {"loss":loss, "softmax": softmax, "labels": y, "confid": pred_confid.squeeze(1)}

        if self.training_stage == 2:
            x, y = batch
            softmax = F.softmax(self.backbone(x), dim=1)
            _, pred_confid = self.network(x)
            pred_confid = torch.sigmoid(pred_confid)
            tcp = softmax.gather(1, y.unsqueeze(1))
            # print("CHECK PRED CONFID", pred_confid.mean(), pred_confid.min(), pred_confid.max())
            # print("CHECK TCP", tcp.mean(), tcp.min(), tcp.max())
            # print(pred_confid[0].item(), y[0].item(), tcp[0].item(), softmax[0])
            loss = F.mse_loss(pred_confid, tcp) # self.loss_mse(pred_confid, tcp) #
            return {"loss":loss, "softmax": softmax, "labels": y, "confid": pred_confid.squeeze(1)}


    # def on_after_backward(self):
    #
    #     if self.global_step % 100 == 0 or 1 == 1:
    #         for ix, x in enumerate(self.backbone.named_parameters()):
    #             if x[1].grad is not None:
    #                 print("GRAD BACKBONE", x[0], x[1].grad.mean())
    #         for ix, x in enumerate(self.network.encoder.named_parameters()):
    #             if x[1].grad is not None:
    #                 print("GRAD CONFID ENCODER", x[0],  x[1].grad.mean())
    #             # if any(x[1].grad):
    #             #     print("CONFID ENCODER GRAD")
    #
    #         for ix, x in enumerate(self.network.confid_net.named_parameters()):
    #             if x[1].grad is not None:
    #                 print("GRAD CONFIDNET",  x[0], x[1].grad.mean())
    #
    #         for ix, x in enumerate(self.named_modules()):
    #             if x[1].training is False:
    #                 print("TRAIN", x[0])


    def validation_step(self, batch, batch_idx):

        if self.training_stage == 0:
            x, y = batch
            logits = self.backbone(x)
            loss = self.loss_ce(logits, y)
            softmax = F.softmax(logits, dim=1)
            return {"loss": loss, "softmax": softmax, "labels": y, "confid": None}

        if self.training_stage == 1:
            x, y = batch
            outputs = self.network(x)
            softmax = F.softmax(outputs[0], dim=1)
            pred_confid = torch.sigmoid(outputs[1])
            tcp = softmax.gather(1, y.unsqueeze(1))
            # print("CHECK PRED CONFID", pred_confid.mean(), pred_confid.min(), pred_confid.max())
            # print("CHECK TCP", tcp.mean(), tcp.min(), tcp.max())
            # print(pred_confid[0].item(), y[0].item(), tcp[0].item(), softmax[0])
            loss = F.mse_loss(pred_confid, tcp)  # self.loss_mse(pred_confid, tcp) #
            return {"loss": loss, "softmax": softmax, "labels": y, "confid": pred_confid.squeeze(1)}

        if self.training_stage == 2:
            x, y = batch
            softmax = F.softmax(self.backbone(x), dim=1)
            _, pred_confid = self.network(x)
            pred_confid = torch.sigmoid(pred_confid)
            tcp = softmax.gather(1, y.unsqueeze(1))
            loss = F.mse_loss(pred_confid, tcp)  # self.loss_mse(pred_confid, tcp) #

            softmax_dist = None
            if self.current_epoch == self.num_epochs - 1:
                # save mcd output for psuedo-test if actual test is with mcd.
                if any("mcd" in cfd for cfd in self.query_confids["test"]):
                    softmax_dist = self.mcd_eval_forward(x=x,
                                                         n_samples=self.monitor_mcd_samples,
                                                         existing_softmax_list=[softmax.unsqueeze(2)])

            return {"loss": loss, "softmax": softmax, "softmax_dist": softmax_dist, "labels": y, "confid": pred_confid.squeeze(1)}


    def test_step(self, batch, batch_idx):
        x, y = batch
        softmax = F.softmax(self.backbone(x), dim=1)
        _, pred_confid = self.network(x)
        pred_confid = torch.sigmoid(pred_confid)

        softmax_dist = None
        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            softmax_dist = self.mcd_eval_forward(x=x,
                                            n_samples=self.test_mcd_samples,
                                            existing_softmax_list=[softmax.unsqueeze(2)])

        self.test_results = {"softmax": softmax, "softmax_dist": softmax_dist, "labels": y, "confid": pred_confid.squeeze(1)}


    def configure_optimizers(self):
        return torch.optim.SGD(self.backbone.parameters(),
                               lr=self.learning_rate,
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)


    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint at epoch {}".format(self.loaded_epoch))




