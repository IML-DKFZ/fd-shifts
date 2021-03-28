import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from src.utils import eval_utils
from src.utils import exp_utils
import numpy as np
import time

class net(pl.LightningModule):

    def __init__(self, cf):
        super(net, self).__init__()

        self.save_hyperparameters()
        self.tensorboard_hparams = eval_utils.get_tb_hparams(cf)
        self.query_monitors = cf.eval.query_monitors
        self.query_confidence_measures = cf.model.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.val_every_n_epoch = cf.trainer.val_every_n_epoch
        self.raw_output_path = cf.exp.raw_output_path
        self.learning_rate = cf.trainer.learning_rate
        self.momentum = cf.trainer.momentum
        self.weight_decay = cf.trainer.weight_decay
        self.num_classes = cf.trainer.num_classes
        self.global_seed = cf.trainer.global_seed
        self.running_correct = []
        self.running_softmax = []
        self.running_labels = []

        self.running_confids = {}
        for k in self.query_confidence_measures:
            self.running_confids[k] = []
        self.brier_score = eval_utils.BrierScore(num_classes=self.num_classes)
        self.loss_criterion = nn.CrossEntropyLoss()

        self.encoder = Encoder(cf)

    def forward(self, x):
        return self.encoder(x)


    def on_train_start(self):
        hp_metrics = {"hp/train_{}".format(k):0 for k in self.query_monitors}
        hp_metrics.update({"hp/val_{}".format(k):0 for k in self.query_monitors})
        self.logger[0].log_hyperparams(self.tensorboard_hparams, hp_metrics)#, {"hp/metric_1": 0, "hp/metric_2": 0})
        exp_utils.set_seed(self.global_seed)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/brier_score", self.brier_score(softmax, y), on_step=False, on_epoch=True)

        tmp_correct = (torch.argmax(softmax, dim=1) == y)*1
        self.running_correct.extend(tmp_correct)

        if "mcp" in self.query_confidence_measures:
            tmp_confids = torch.max(softmax, dim=1)[0]
            self.running_confids["mcp"].extend(tmp_confids)
        if "pe" in self.query_confidence_measures:
            tmp_confids = torch.sum(softmax * (- torch.log(softmax)), dim=1)
            self.running_confids["pe"].extend(tmp_confids)

        return loss # this will be "outputs" later, what exactly can I return here?

    def training_epoch_end(self, outputs):
        # optinally perform at random step with if self.global_step ... and set on_step=True. + reset!
        # the whole thing takes 0.3 sec atm.
        do_plot = True if (self.current_epoch + 1) % self.val_every_n_epoch == 0 else False
        monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confids,
                                                                 self.running_correct,
                                                                 self.query_monitors,
                                                                 do_plot = do_plot
                                                                 )

        tensorboard = self.logger[0].experiment
        self.log("step", self.current_epoch)
        for k, v in monitor_metrics.items():
            self.log("train/{}".format(k), v)
            tensorboard.add_scalar("hp/train_{}".format(k), v, global_step=self.current_epoch)

        if do_plot:
            for k,v in monitor_plots.items():
                tensorboard.add_figure("train/{}".format(k), v, self.current_epoch)

        self.running_correct = []
        self.running_confids = {}
        for k in self.query_confidence_measures:
            self.running_confids[k] = []


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/brier_score", self.brier_score(softmax, y), on_step=False, on_epoch=True)

        tmp_correct = (torch.argmax(softmax, dim=1) == y) * 1
        self.running_correct.extend(tmp_correct)

        if "mcp" in self.query_confidence_measures:
            tmp_confids = torch.max(softmax, dim=1)[0]
            self.running_confids["mcp"].extend(tmp_confids)
        if "pe" in self.query_confidence_measures:
            tmp_confids = torch.sum(softmax * (- torch.log(softmax)), dim=1)
            self.running_confids["pe"].extend(tmp_confids)

        if self.current_epoch == self.num_epochs -1:
            self.running_softmax.extend(softmax)
            self.running_labels.extend(y)

        return loss


    def validation_epoch_end(self, outputs):
        monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confids,
                                                                 self.running_correct,
                                                                 self.query_monitors)

        tensorboard = self.logger[0].experiment
        self.log("step", self.current_epoch)
        for k, v in monitor_metrics.items():
            self.log("val/{}".format(k), v)
            tensorboard.add_scalar("hp/val_{}".format(k), v, global_step=self.current_epoch)

        for k,v in monitor_plots.items():
            tensorboard.add_figure("val/{}".format(k), v, self.current_epoch)

        self.running_correct = []
        self.running_confids = {}
        for k in self.query_confidence_measures:
            self.running_confids[k] = []


    def on_fit_end(self):
        eval_utils.clean_logging(self.logger[1].experiment.log_dir)
        raw_output = torch.cat([torch.stack(self.running_softmax, dim=0),
                                  torch.stack(self.running_labels, dim=0).unsqueeze(1)]
                                 , dim=1)
        np.save(self.raw_output_path, raw_output.cpu().data.numpy())


    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint at epoch {}".format(self.loaded_epoch))


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        tmp_correct = (torch.argmax(softmax, dim=1) == y) * 1
        self.running_correct = self.running_correct.append(tmp_correct)

        if "mcp" in self.query_confidence_measures:
            tmp_confids = torch.max(softmax, dim=1)[0]
            self.running_confids["mcp"] = torch.cat([self.running_confids["mcp"], tmp_confids])
        if "pe" in self.query_confidence_measures:
            tmp_confids = torch.sum(softmax * (- torch.log(softmax)), dim=1)
            self.running_confids["pe"] = torch.cat([self.running_confids["pe"], tmp_confids])

    def on_test_end(self):

        monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confids,
                                                                 self.running_correct,
                                                                 self.query_monitors)

        monitor_metrics.update({"test_epoch": self.loaded_epoch})
        self.logger.log_metrics(monitor_metrics)
        self.logger.save()
        tensorboard = self.logger[0].experiment
        for k, v in monitor_plots.items():
            tensorboard.add_figure("test/{}".format(k), v, self.current_epoch)


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.learning_rate,
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)



class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, cf):
        super(Encoder, self).__init__()

        self.img_size = cf.data.img_size
        self.fc_dim = cf.model.fc_dim
        self.num_classes = cf.trainer.num_classes
        self.confidence_mode = cf.model.method
        self.dropout_rate = 0.3

        self.conv1 = Conv2dSame(self.img_size[-1], 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSame(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.conv3 = Conv2dSame(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = Conv2dSame(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.conv5 = Conv2dSame(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = Conv2dSame(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.fc1 = nn.Linear(2048, self.fc_dim)
        self.dropout4 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.maxpool1(x)
        if self.confidence_mode == "mcdropout":
            x = F.dropout(x, self.dropout_rate, training=self.training)
        else:
            x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = F.relu(self.conv4(x))
        x = self.conv4_bn(x)
        x = self.maxpool2(x)
        if self.confidence_mode == "mcdropout":
            x = F.dropout(x, self.dropout_rate, training=self.training)
        else:
            x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.conv5_bn(x)
        x = F.relu(self.conv6(x))
        x = self.conv6_bn(x)
        x = self.maxpool3(x)
        if self.confidence_mode == "mcdropout":
            x = F.dropout(x, self.dropout_rate, training=self.training)
        else:
            x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.confidence_mode == "mcdropout":
            x = F.dropout(x, self.dropout_rate, training=self.training)
        else:
            x = self.dropout4(x)
        x = self.fc2(x)
        return x

