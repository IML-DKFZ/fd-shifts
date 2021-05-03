from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.connectors.logger_connector.logger_connector import LoggerConnector
import torch
import numpy as np
from src.utils import eval_utils


class ConfidMonitor(Callback):

    def __init__(self, cf):

        self.num_epochs = cf.trainer.num_epochs
        self.num_classes = cf.data.num_classes
        self.fast_dev_run = cf.trainer.fast_dev_run


        self.tensorboard_hparams = eval_utils.get_tb_hparams(cf)
        self.query_performance_metrics = cf.eval.performance_metrics
        self.query_confid_metrics = cf.eval.confid_metrics
        self.query_monitor_plots = cf.eval.monitor_plots
        self.query_confids = cf.eval.confidence_measures

        self.raw_output_path_fit = cf.exp.raw_output_path
        self.external_confids_output_path_fit = cf.exp.external_confids_output_path
        self.raw_output_path_test = cf.test.raw_output_path
        self.external_confids_output_path_test = cf.test.external_confids_output_path
        self.version_dir = cf.exp.version_dir
        self.val_every_n_epoch = cf.trainer.val_every_n_epoch

        self.running_test_softmax = []
        self.running_test_labels = []
        self.running_test_dataset_idx = []
        self.running_test_external_confids = []
        self.running_confid_stats = {}
        self.running_perf_stats = {}
        self.running_confid_stats["train"] = {k: {"confids": [], "correct": []} for k in self.query_confids["train"]}
        self.running_confid_stats["val"] = {k: {"confids": [], "correct": []} for k in self.query_confids["val"]}
        self.running_train_correct_sum_sanity = 0
        self.running_val_correct_sum_sanity = 0
        self.running_perf_stats["train"] = {k: [] for k in self.query_performance_metrics["train"]}
        self.running_perf_stats["val"] = {k: [] for k in self.query_performance_metrics["val"]}




    def on_train_start(self, trainer, pl_module):
        if self.fast_dev_run is False:
            hp_metrics = {"hp/train_{}".format(k):0 for k in self.query_performance_metrics}
            hp_metrics.update({"hp/val_{}".format(k):0 for k in self.query_performance_metrics})
            pl_module.logger[0].log_hyperparams(self.tensorboard_hparams, hp_metrics)#, {"hp/metric_1": 0, "hp/metric_2": 0})


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        # outputs keys meta/extra/minimze
        loss = outputs[0][0]["minimize"]
        softmax = outputs[0][0]["extra"]["softmax"]
        y = outputs[0][0]["extra"]["labels"]

        tmp_correct = None
        if len(self.running_perf_stats["train"].keys()) > 0:
            stat_keys = self.running_perf_stats["train"].keys()
            y_one_hot = None
            if "loss" in stat_keys:
                self.running_perf_stats["train"]["loss"].append(loss)
            if "nll" in stat_keys:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
                self.running_perf_stats["train"]["nll"].append(torch.sum(-torch.log(softmax + 1e-7)*y_one_hot, dim=1).mean())
            if "accuracy" in stat_keys:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
                self.running_perf_stats["train"]["accuracy"].append(tmp_correct.sum() / tmp_correct.numel())
            if "brier_score" in stat_keys:
                if y_one_hot is None:
                    y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
                self.running_perf_stats["train"]["brier_score"].append(((softmax - y_one_hot) ** 2).sum(1).mean())

        if len(self.running_confid_stats["train"].keys()) > 0:
            self.running_train_correct_sum_sanity += tmp_correct.sum()
            stat_keys = self.running_confid_stats["train"].keys()
            if tmp_correct is None:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
            if "det_mcp" in stat_keys:
                tmp_confids = torch.max(softmax, dim=1)[0]
                self.running_confid_stats["train"]["det_mcp"]["confids"].extend(tmp_confids)
                self.running_confid_stats["train"]["det_mcp"]["correct"].extend(tmp_correct)
            if "det_pe" in stat_keys:
                tmp_confids = torch.sum(softmax * (- torch.log(softmax + 1e-7)), dim=1)
                self.running_confid_stats["train"]["det_pe"]["confids"].extend(tmp_confids)
                self.running_confid_stats["train"]["det_pe"]["correct"].extend(tmp_correct)

            if "ext" in stat_keys:
                tmp_confids = outputs[0][0]["extra"]["confid"]
                if tmp_confids is not None:
                    self.running_confid_stats["train"]["ext"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["train"]["ext"]["correct"].extend(tmp_correct)


    def on_train_epoch_end(self, trainer, pl_module, outputs):

        if (len(self.running_confid_stats["train"].keys()) > 0 or len(self.running_perf_stats["train"].keys()) > 0)\
                and (self.running_train_correct_sum_sanity > 0):
            do_plot = True if (pl_module.current_epoch + 1) % self.val_every_n_epoch == 0 and \
                              len(self.query_confids["train"]) > 0 and len(self.query_monitor_plots) > 0 else False
            monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confid_stats["train"],
                                                                     self.running_perf_stats["train"],
                                                                     self.query_confid_metrics["train"],
                                                                     self.query_monitor_plots,
                                                                     do_plot = do_plot,
                                                                     ext_confid_name=pl_module.ext_confid_name
                                                                     )
            print("CHECK TRAIN METRICS", monitor_metrics)
            tensorboard = pl_module.logger[0].experiment
            pl_module.log("step", pl_module.current_epoch)
            for k, v in monitor_metrics.items():
                pl_module.log("train/{}".format(k), v)
                tensorboard.add_scalar("hp/train_{}".format(k), v, global_step=pl_module.current_epoch)

            if do_plot:
                for k, v in monitor_plots.items():
                    tensorboard.add_figure("train/{}".format(k), v, pl_module.current_epoch)

        self.running_confid_stats["train"] = {k: {"confids": [], "correct": []} for k in
                                              self.query_confids["train"]}
        self.running_perf_stats["train"] = {k: [] for k in self.query_performance_metrics["train"]}
        self.running_train_correct_sum_sanity = 0


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        # todo organize in dicts / factories instead of if else statements?

        tmp_correct = None
        loss = outputs["loss"]
        softmax = outputs["softmax"]
        y = outputs["labels"]
        softmax_dist = outputs.get("softmax_dist")


        if len(self.running_perf_stats["val"].keys()) > 0:
            stat_keys = self.running_perf_stats["val"].keys()
            y_one_hot = None
            if "loss" in stat_keys:
                self.running_perf_stats["val"]["loss"].append(loss)
            if "nll" in stat_keys:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
                self.running_perf_stats["val"]["nll"].append(torch.sum(-torch.log(softmax + 1e-7) * y_one_hot, dim=1).mean())
            if "accuracy" in stat_keys:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
                # print(tmp_correct.sum())
                self.running_perf_stats["val"]["accuracy"].append(tmp_correct.sum() / tmp_correct.numel())
            if "brier_score" in stat_keys:
                if y_one_hot is None:
                    y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
                self.running_perf_stats["val"]["brier_score"].append(((softmax - y_one_hot) ** 2).sum(1).mean())

        if len(self.running_confid_stats["val"].keys()) > 0 or softmax_dist is not None:

            stat_keys = self.running_confid_stats["val"].keys()
            if tmp_correct is None:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.cuda.ByteTensor)
            self.running_val_correct_sum_sanity += tmp_correct.sum()
            if "det_mcp" in stat_keys:
                tmp_confids = torch.max(softmax, dim=1)[0]
                self.running_confid_stats["val"]["det_mcp"]["confids"].extend(tmp_confids)
                self.running_confid_stats["val"]["det_mcp"]["correct"].extend(tmp_correct)
            if "det_pe" in stat_keys:
                tmp_confids = torch.sum(softmax * (- torch.log(softmax + 1e-7)), dim=1)
                self.running_confid_stats["val"]["det_pe"]["confids"].extend(tmp_confids)
                self.running_confid_stats["val"]["det_pe"]["correct"].extend(tmp_correct)

            if "ext" in stat_keys:
                tmp_confids = outputs["confid"]
                if tmp_confids is not None:
                    self.running_confid_stats["val"]["ext"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["val"]["ext"]["correct"].extend(tmp_correct)



            if softmax_dist is not None:

                mean_softmax = torch.mean(softmax_dist, dim=2)
                tmp_mcd_correct = (torch.argmax(mean_softmax, dim=1) == y).type(torch.cuda.ByteTensor)

                if "mcd_mcp" in stat_keys:
                    tmp_confids = torch.max(mean_softmax, dim=1)[0]
                    self.running_confid_stats["val"]["mcd_mcp"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["val"]["mcd_mcp"]["correct"].extend(tmp_mcd_correct)
                if "mcd_pe" in stat_keys:
                    pe_confids = torch.sum(mean_softmax * (- torch.log(mean_softmax + 1e-7)), dim=1)
                    self.running_confid_stats["val"]["mcd_pe"]["confids"].extend(pe_confids)
                    self.running_confid_stats["val"]["mcd_pe"]["correct"].extend(tmp_mcd_correct)
                if "mcd_ee" in stat_keys:
                    ee_confids = torch.sum(softmax_dist * (- torch.log(softmax_dist + 1e-7)), dim=1).mean(1)
                    self.running_confid_stats["val"]["mcd_ee"]["confids"].extend(ee_confids)
                    self.running_confid_stats["val"]["mcd_ee"]["correct"].extend(tmp_mcd_correct)
                if "mcd_mi" in stat_keys:
                    tmp_confids = pe_confids - ee_confids
                    self.running_confid_stats["val"]["mcd_mi"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["val"]["mcd_mi"]["correct"].extend(tmp_mcd_correct)
                if "mcd_sv" in stat_keys:
                    tmp_confids = ((softmax_dist - mean_softmax.unsqueeze(2)) ** 2).mean((1, 2))
                    self.running_confid_stats["val"]["mcd_sv"]["confids"].extend(tmp_confids)
                    self.running_confid_stats["val"]["mcd_sv"]["correct"].extend(tmp_mcd_correct)

        if pl_module.current_epoch == self.num_epochs - 1:
            self.running_test_softmax.extend(softmax_dist if softmax_dist is not None else softmax)
            self.running_test_labels.extend(y)
            if "ext" in self.running_confid_stats["val"].keys():
                self.running_test_external_confids.extend(outputs["confid"])


    def on_validation_epoch_end(self, trainer, pl_module):

        monitor_metrics = None
        if (len(self.running_confid_stats["val"].keys()) > 0 or len(self.running_perf_stats["val"].keys()) > 0) \
                and (self.running_val_correct_sum_sanity > 0):
            do_plot = True if len(self.query_confids["val"]) > 0 and len(self.query_monitor_plots) > 0 else False
            monitor_metrics, monitor_plots = eval_utils.monitor_eval(self.running_confid_stats["val"],
                                                                     self.running_perf_stats["val"],
                                                                     self.query_confid_metrics["val"],
                                                                     self.query_monitor_plots,
                                                                     do_plot=do_plot,
                                                                     ext_confid_name=pl_module.ext_confid_name
                                                                     )
            tensorboard = pl_module.logger[0].experiment
            pl_module.log("step", pl_module.current_epoch)
            for k, v in monitor_metrics.items():
                pl_module.log("val/{}".format(k), v)
                tensorboard.add_scalar("hp/val_{}".format(k), v, global_step=pl_module.current_epoch)

            if do_plot:
                for k, v in monitor_plots.items():
                    tensorboard.add_figure("val/{}".format(k), v, pl_module.current_epoch)

        print("CHECK VAL METRICS", monitor_metrics)
        if hasattr(pl_module, "selection_metrics"):
            for metric, mode in zip(pl_module.selection_metrics, pl_module.selection_modes):
                if monitor_metrics is None or metric.split("/")[-1] not in monitor_metrics.keys():
                    dummy = 0 if mode == "max" else 1
                    print("selection metric {} not computed, replacing with {}.".format(metric, dummy))
                    pl_module.log("{}".format(metric), dummy)

        self.running_confid_stats["val"] = {k: {"confids": [], "correct": []} for k in
                                            self.query_confids["val"]}
        self.running_perf_stats["val"] = {k: [] for k in self.query_performance_metrics["val"]}
        self.running_val_correct_sum_sanity = 0


    def on_train_end(self, trainer, pl_module):
        if len(self.running_test_softmax) > 0:
            stacked_softmax = torch.stack(self.running_test_softmax, dim=0)
            stacked_labels = torch.stack(self.running_test_labels, dim=0).unsqueeze(1)
            stacked_dataset_idx = torch.zeros_like(stacked_labels)
            raw_output = torch.cat([stacked_softmax.reshape(stacked_softmax.size()[0], -1),
                                    stacked_labels, stacked_dataset_idx],
                                   dim=1)
            np.save(self.raw_output_path_fit, raw_output.cpu().data.numpy())
            print("saved raw validation outputs to {}".format(self.raw_output_path_fit))

            if "ext" in self.query_confids["test"]:
                stacked_external_confids = torch.stack(self.running_test_external_confids, dim=0)
                np.save(self.external_confids_output_path_fit, stacked_external_confids.cpu().data.numpy())
                print("saved external confids validation outputs to {}".format(self.external_confids_output_path_fit))

            self.running_test_softmax = []
            self.running_test_labels = []
            self.running_test_external_confids = []

        eval_utils.clean_logging(self.version_dir)


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        outputs = pl_module.test_results

        self.running_test_softmax.extend(outputs["softmax"].to(dtype=torch.float16).cpu())
        self.running_test_labels.extend(outputs["labels"].cpu())
        if "ext" in self.query_confids["test"]:
            self.running_test_external_confids.extend(outputs["confid"].to(dtype=torch.float32).cpu())

        self.running_test_dataset_idx.extend(torch.ones_like(outputs["labels"].cpu()) * dataloader_idx)


    def on_test_end(self, trainer, pl_module):

        stacked_softmax = torch.stack(self.running_test_softmax, dim=0)
        stacked_labels = torch.stack(self.running_test_labels, dim=0).unsqueeze(1)
        stacked_dataset_idx = torch.stack(self.running_test_dataset_idx, dim=0).unsqueeze(1)
        raw_output = torch.cat([stacked_softmax.reshape(stacked_softmax.size()[0], -1),
                                stacked_labels, stacked_dataset_idx],
                               dim=1)

        np.save(self.raw_output_path_test, raw_output.cpu().data.numpy())
        print("saved raw test outputs to {}".format(self.raw_output_path_test))

        if "ext" in self.query_confids["test"]:
            stacked_external_confids = torch.stack(self.running_test_external_confids, dim=0)
            np.save(self.external_confids_output_path_test, stacked_external_confids.cpu().data.numpy())



