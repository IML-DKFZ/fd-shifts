

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from utils import eval_utils
import time


class net(pl.LightningModule):
#class net(nn.Module):

    def __init__(self, cf):
        super(net, self).__init__()

        self.query_monitors = cf.eval.query_monitors
        self.learning_rate = cf.trainer.learning_rate
        self.momentum = cf.trainer.momentum
        self.weight_decay = cf.trainer.weight_decay
        self.num_classes = cf.trainer.num_classes

        self.running_correct = torch.Tensor([]).long().cuda()
        self.running_confids = torch.Tensor([]).cuda()
        self.brier_score = eval_utils.BrierScore(num_classes=self.num_classes)
        self.loss_criterion = nn.CrossEntropyLoss()

        self.encoder = Encoder(cf)

    def forward(self, x):
        return self.encoder(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/brier_score", self.brier_score(softmax, y), on_step=False, on_epoch=True)

        tmp_confids = torch.max(softmax, dim=1)[0]
        tmp_correct = (torch.argmax(softmax, dim=1) == y)*1

        self.running_confids = torch.cat([self.running_confids, tmp_confids])
        self.running_correct = torch.cat([self.running_correct, tmp_correct])

        return loss # this will be "outputs" later, what exactly can I return here?

    def training_epoch_end(self, outputs):
        # optinally perform at random step with if self.global_step ... and set on_step=True. + reset!
        # the whole thing takes 0.3 sec atm.
        monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confids,
                                                                 self.running_correct,
                                                                 self.query_monitors)

        for k, v in monitor_metrics.items():
            self.log("train/{}".format(k), v)
        self.log("step", self.current_epoch)

        tensorboard = self.logger[0].experiment
        for k,v in monitor_plots.items():
            tensorboard.add_figure("train/{}".format(k), v, self.current_epoch)

        self.running_correct = torch.Tensor([]).long().cuda()
        self.running_confids = torch.Tensor([]).cuda()
        self.running_brier_score = torch.Tensor([]).cuda()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/brier_score", self.brier_score(softmax, y), on_step=False, on_epoch=True)

        tmp_confids = torch.max(softmax, dim=1)[0]
        tmp_correct = (torch.argmax(softmax, dim=1) == y) * 1

        self.running_confids = torch.cat([self.running_confids, tmp_confids])
        self.running_correct = torch.cat([self.running_correct, tmp_correct])

        return loss


    def validation_epoch_end(self, outputs):

        monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confids,
                                                                 self.running_correct,
                                                                 self.query_monitors)

        for k, v in monitor_metrics.items():
            self.log("val/{}".format(k), v)
        self.log("step", self.current_epoch)

        tensorboard = self.logger[0].experiment
        for k, v in monitor_plots.items():
            tensorboard.add_figure("val/{}".format(k), v, self.current_epoch)

        self.running_correct = torch.Tensor([]).long().cuda()
        self.running_confids = torch.Tensor([]).cuda()
        self.running_brier_score = torch.Tensor([]).cuda()


    def on_train_end(self):
        eval_utils.clean_logging(self.logger[1].experiment.log_dir)


    def test_step(self, batch, batch_idx):
        pass
        #     x, y = batch
        #     y_hat = self(x)
        #     loss = F.cross_entropy(y_hat, y)
        # #    self.log('test_loss', loss)
        #     return loss

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

        self.img_size = cf.exp.img_size
        self.fc_dim = cf.model.fc_dim
        self.num_classes = cf.trainer.num_classes
        self.confidence_mode = cf.exp.confidence_mode

        self.conv1 = Conv2dSame(self.img_size[-1], 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSame(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = Conv2dSame(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = Conv2dSame(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv5 = Conv2dSame(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = Conv2dSame(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(2048, self.fc_dim)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.maxpool1(x)
        if self.confidence_mode == "mcd":
            x = F.dropout(x, 0.3, training=self.training)
        else:
            x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = F.relu(self.conv4(x))
        x = self.conv4_bn(x)
        x = self.maxpool2(x)
        if self.confidence_mode == "mcd":
            x = F.dropout(x, 0.3, training=self.training)
        else:
            x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.conv5_bn(x)
        x = F.relu(self.conv6(x))
        x = self.conv6_bn(x)
        x = self.maxpool3(x)
        if self.confidence_mode == "mcd":
            x = F.dropout(x, 0.3, training=self.training)
        else:
            x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.confidence_mode == "mcd":
            x = F.dropout(x, 0.3, training=self.training)
        else:
            x = self.dropout4(x)
        x = self.fc2(x)
        return x