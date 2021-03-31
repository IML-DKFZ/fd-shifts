import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from src.utils import eval_utils
from src.utils import exp_utils
import numpy as np

class net(pl.LightningModule):

    def __init__(self, cf):
        super(net, self).__init__()

        self.save_hyperparameters()
        self.tensorboard_hparams = eval_utils.get_tb_hparams(cf)
        self.query_performance_metrics = cf.eval.performance_metrics
        self.query_confid_metrics = cf.eval.confid_metrics
        self.query_monitor_plots = cf.eval.monitor_plots
        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.fast_dev_run = cf.trainer.fast_dev_run

        self.monitor_mcd_samples = cf.model.monitor_mcd_samples
        self.test_mcd_samples = cf.model.test_mcd_samples

        self.val_every_n_epoch = cf.trainer.val_every_n_epoch
        self.raw_output_path_fit = cf.exp.raw_output_path
        self.raw_output_path_test = cf.test.raw_output_path
        self.learning_rate = cf.trainer.learning_rate
        self.momentum = cf.trainer.momentum
        self.weight_decay = cf.trainer.weight_decay
        self.num_classes = cf.trainer.num_classes
        self.global_seed = cf.trainer.global_seed

        self.running_softmax = []
        self.running_labels = []
        self.running_confid_stats = {}
        self.running_perf_stats = {}
        self.running_confid_stats["train"] = {k: {"confids":[], "correct":[]} for k in self.query_confids["train"]}
        self.running_confid_stats["val"] = {k: {"confids":[], "correct":[]} for k in self.query_confids["val"]}
        self.running_perf_stats["train"] = {k:[] for k in self.query_performance_metrics["train"]}
        self.running_perf_stats["val"] = {k:[] for k in self.query_performance_metrics["val"]}

        self.loss_criterion = nn.CrossEntropyLoss()

        self.encoder = Encoder(cf)

    def forward(self, x):
        return self.encoder(x)


    def mcd_eval_forward(self, x, n_samples, existing_softmax_list=None):
        self.encoder.eval_mcdropout = True
        softmax_list = existing_softmax_list if existing_softmax_list is not None else []
        for _ in range(n_samples - len(softmax_list)):
            softmax_list.append(F.softmax(self.encoder(x), dim=1).unsqueeze(2))
        self.encoder.eval_mcdropout = False

        return torch.cat(softmax_list, dim=2)


    def on_train_start(self):
        if self.fast_dev_run is False:
            hp_metrics = {"hp/train_{}".format(k):0 for k in self.query_performance_metrics}
            hp_metrics.update({"hp/val_{}".format(k):0 for k in self.query_performance_metrics})
            self.logger[0].log_hyperparams(self.tensorboard_hparams, hp_metrics)#, {"hp/metric_1": 0, "hp/metric_2": 0})
        exp_utils.set_seed(self.global_seed)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        tmp_correct = None
        if len(self.running_perf_stats["train"].keys()) > 0:
            stat_keys = self.running_perf_stats["train"].keys()
            if "nll" in stat_keys:
                self.running_perf_stats["train"]["nll"].append(loss)
            if "accuracy" in stat_keys:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
                self.running_perf_stats["train"]["accuracy"].append(tmp_correct.sum()/tmp_correct.numel())
            if "brier_score" in stat_keys:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
                self.running_perf_stats["train"]["brier_score"].append(((softmax - y_one_hot) ** 2).sum(1).mean())

        if len(self.running_confid_stats["train"].keys()) > 0:
            stat_keys = self.running_confid_stats["train"].keys()
            if tmp_correct is None:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
            if "det_mcp" in stat_keys:
                tmp_confids = torch.max(softmax, dim=1)[0]
                self.running_confid_stats["train"]["det_mcp"]["confids"].extend(tmp_confids)
                self.running_confid_stats["train"]["det_mcp"]["correct"].extend(tmp_correct)
            if "det_pe" in stat_keys:
                tmp_confids = torch.sum(softmax * (- torch.log(softmax)), dim=1)
                self.running_confid_stats["train"]["det_pe"]["confids"].extend(tmp_confids)
                self.running_confid_stats["train"]["det_pe"]["correct"].extend(tmp_correct)

        return loss

    def training_epoch_end(self, outputs):
        if len(self.running_confid_stats["train"].keys()) > 0 or len(self.running_perf_stats["train"].keys()) > 0:
            do_plot = True if (self.current_epoch + 1) % self.val_every_n_epoch == 0 and \
                              len(self.query_confids["train"]) > 0 and len(self.query_monitor_plots) > 0 else False
            monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confid_stats["train"],
                                                                     self.running_perf_stats["train"],
                                                                     self.query_confid_metrics["train"],
                                                                     self.query_monitor_plots,
                                                                     do_plot = do_plot
                                                                     )

            tensorboard = self.logger[0].experiment
            self.log("step", self.current_epoch)
            for k, v in monitor_metrics.items():
                self.log("train/{}".format(k), v)
                tensorboard.add_scalar("hp/train_{}".format(k), v, global_step=self.current_epoch)

            if do_plot:
                for k, v in monitor_plots.items():
                    tensorboard.add_figure("train/{}".format(k), v, self.current_epoch)

            self.running_confid_stats["train"] = {k: {"confids": [], "correct": []} for k in
                                                  self.query_confids["train"]}
            self.running_perf_stats["train"] = {k: [] for k in self.query_performance_metrics["train"]}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        loss = self.loss_criterion(logits, y)
        softmax = F.softmax(logits, dim=1)

        tmp_correct = None

        if len(self.running_perf_stats["val"].keys()) > 0:
            stat_keys = self.running_perf_stats["val"].keys()
            if "nll" in stat_keys:
                self.running_perf_stats["val"]["nll"].append(loss)
            if "accuracy" in stat_keys:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
                self.running_perf_stats["val"]["accuracy"].append(tmp_correct.sum() / tmp_correct.numel())
            if "brier_score" in stat_keys:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
                self.running_perf_stats["val"]["brier_score"].append(((softmax - y_one_hot) ** 2).sum(1).mean())

        if len(self.running_confid_stats["val"].keys()) > 0:
            stat_keys = self.running_confid_stats["val"].keys()
            if tmp_correct is None:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
            if "det_mcp" in stat_keys:
                tmp_confids = torch.max(softmax, dim=1)[0]
                self.running_confid_stats["val"]["det_mcp"]["confids"].extend(tmp_confids)
                self.running_confid_stats["val"]["det_mcp"]["correct"].extend(tmp_correct)
            if "det_pe" in stat_keys:
                tmp_confids = torch.sum(softmax * (- torch.log(softmax)), dim=1)
                self.running_confid_stats["val"]["det_pe"]["confids"].extend(tmp_confids)
                self.running_confid_stats["val"]["det_pe"]["correct"].extend(tmp_correct)


            softmax_dist = None
            if any("mcd" in cfd for cfd in stat_keys):
                softmax_dist = self.mcd_eval_forward(x=x,
                                                     n_samples=self.monitor_mcd_samples,
                                                     existing_softmax_list=[softmax.unsqueeze(2)])
                mean_softmax = torch.mean(softmax_dist, dim=2)
                tmp_mcd_correct = (torch.argmax(mean_softmax, dim=1) == y).type(torch.cuda.ByteTensor)

                if "mcd_mcp" in stat_keys:
                    tmp_confids = torch.max(mean_softmax, dim=1)[0]
                    self.running_confid_stats["val"]["mcd_mcp"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["val"]["mcd_mcp"]["correct"].extend(tmp_mcd_correct)
                if "mcd_pe" in stat_keys:
                    pe_confids = torch.sum(mean_softmax * (- torch.log(mean_softmax)), dim=1)
                    self.running_confid_stats["val"]["mcd_pe"]["confids"].extend(pe_confids)
                    self.running_confid_stats["val"]["mcd_pe"]["correct"].extend(tmp_mcd_correct)
                if "mcd_ee" in stat_keys:
                    ee_confids = torch.sum(softmax_dist * (- torch.log(softmax_dist)), dim=1).mean(1)
                    self.running_confid_stats["val"]["mcd_ee"]["confids"].extend(ee_confids)
                    self.running_confid_stats["val"]["mcd_ee"]["correct"].extend(tmp_mcd_correct)
                if "mcd_mi" in stat_keys:
                    tmp_confids = pe_confids - ee_confids
                    self.running_confid_stats["val"]["mcd_mi"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["val"]["mcd_mi"]["correct"].extend(tmp_mcd_correct)
                if "mcd_sv" in stat_keys:
                    tmp_confids = ((softmax_dist - mean_softmax.unsqueeze(2))**2).mean((1, 2))
                    self.running_confid_stats["val"]["mcd_sv"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["val"]["mcd_sv"]["correct"].extend(tmp_mcd_correct)

        if self.current_epoch == self.num_epochs -1:
            # save mcd output for psuedo-test if actual test is with mcd.
            if any("mcd" in cfd for cfd in self.query_confids["test"]):
                if softmax_dist is None:
                    softmax_dist = self.mcd_eval_forward(x=x,
                                                         n_samples=self.monitor_mcd_samples,
                                                         existing_softmax_list=[softmax.unsqueeze(2)])
                out_softmax = softmax_dist
            else:
                out_softmax = softmax
            self.running_softmax.extend(out_softmax)
            self.running_labels.extend(y)

        return loss


    def validation_epoch_end(self, outputs):
        if len(self.running_confid_stats["val"].keys()) > 0 or len(self.running_perf_stats["val"].keys()) > 0:
            do_plot = True if len(self.query_confids["val"]) > 0 and len(self.query_monitor_plots) > 0 else False
            monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confid_stats["val"],
                                                                     self.running_perf_stats["val"],
                                                                     self.query_confid_metrics["val"],
                                                                     self.query_monitor_plots,
                                                                     do_plot=do_plot
                                                                     )
            tensorboard = self.logger[0].experiment
            self.log("step", self.current_epoch)
            for k, v in monitor_metrics.items():
                self.log("val/{}".format(k), v)
                tensorboard.add_scalar("hp/val_{}".format(k), v, global_step=self.current_epoch)

            if do_plot:
                for k, v in monitor_plots.items():
                    tensorboard.add_figure("val/{}".format(k), v, self.current_epoch)

            self.running_confid_stats["val"] = {k: {"confids": [], "correct": []} for k in
                                                  self.query_confids["val"]}
            self.running_perf_stats["val"] = {k: [] for k in self.query_performance_metrics["val"]}


    def on_train_end(self):
        if len(self.running_softmax) > 0:
            stacked_softmax = torch.stack(self.running_softmax, dim=0)
            stacked_labels = torch.stack(self.running_labels, dim=0).unsqueeze(1)
            raw_output = torch.cat([stacked_softmax.reshape(stacked_softmax.size()[0], -1),
                                    stacked_labels],
                                    dim=1)
            np.save(self.raw_output_path_fit, raw_output.cpu().data.numpy())
            print("saved raw validation outputs to {}".format(self.raw_output_path_fit))

            self.running_softmax = []
            self.running_labels = []


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.encoder(x)
        softmax = F.softmax(logits, dim=1)

        if any("mcd" in cfd for cfd in self.query_confids["test"]):
            out_softmax = self.mcd_eval_forward(x=x,
                                                n_samples=self.test_mcd_samples,
                                                existing_softmax_list=[softmax.unsqueeze(2)])
        else:
            out_softmax = softmax
        self.running_softmax.extend(out_softmax)
        self.running_labels.extend(y)


    def on_test_end(self):

        stacked_softmax = torch.stack(self.running_softmax, dim=0)
        stacked_labels = torch.stack(self.running_labels, dim=0).unsqueeze(1)
        raw_output = torch.cat([stacked_softmax.reshape(stacked_softmax.size()[0], -1),
                                stacked_labels],
                               dim=1)

        np.save(self.raw_output_path_test, raw_output.cpu().data.numpy())
        print("saved raw test outputs to {}".format(self.raw_output_path_test))


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.learning_rate,
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)


    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        print("loading checkpoint at epoch {}".format(self.loaded_epoch))



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
        self.dropout_rate = 0.3
        self.eval_mcdropout = False

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
        if self.eval_mcdropout:
            x = F.dropout(x, self.dropout_rate, training=True)
        else:
            x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = F.relu(self.conv4(x))
        x = self.conv4_bn(x)
        x = self.maxpool2(x)
        if self.eval_mcdropout:
            x = F.dropout(x, self.dropout_rate, training=True)
        else:
            x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.conv5_bn(x)
        x = F.relu(self.conv6(x))
        x = self.conv6_bn(x)
        x = self.maxpool3(x)
        if self.eval_mcdropout:
            x = F.dropout(x, self.dropout_rate, training=True)
        else:
            x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.eval_mcdropout:
            x = F.dropout(x, self.dropout_rate, training=True)
        else:
            x = self.dropout4(x)
        x = self.fc2(x)
        return x

