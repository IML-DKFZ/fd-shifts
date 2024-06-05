import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.connectors.logger_connector.logger_connector import (
    LoggerConnector,
)
from rich import print
from tqdm import tqdm

from fd_shifts import configs, logger
from fd_shifts.analysis import eval_utils

DTYPES = {
    16: torch.float16,
    32: torch.float16,
    64: torch.float16,
}


class ConfidMonitor(Callback):
    """Callback to save confids and logits to disk

    Attributes:
        sync_dist:
        cfg:
        output_dtype:
        num_epochs:
        num_classes:
        fast_dev_run:
        tensorboard_hparams:
        query_performance_metrics:
        query_confid_metrics:
        query_monitor_plots:
        query_confids:
        output_paths:
        version_dir:
        val_every_n_epoch:
        running_test_softmax:
        running_test_softmax_dist:
        running_test_labels:
        running_test_dataset_idx:
        running_test_external_confids:
        running_test_external_confids_dist:
        running_confid_stats:
        running_perf_stats:
        running_train_correct_sum_sanity:
        running_val_correct_sum_sanity:
        running_train_correct_sum_sanity:
        running_val_correct_sum_sanity:
    """

    def __init__(self, cf: configs.Config):
        self.sync_dist = True if torch.cuda.device_count() > 1 else False

        self.cfg = cf
        self.output_dtype = DTYPES[cf.test.output_precision]

        self.num_epochs = cf.trainer.num_epochs
        if self.num_epochs is None:
            self.num_epochs = cf.trainer.num_steps
        self.num_classes = cf.data.num_classes
        self.fast_dev_run = cf.trainer.fast_dev_run

        self.tensorboard_hparams = eval_utils._get_tb_hparams(cf)
        self.query_performance_metrics = cf.eval.performance_metrics
        self.query_confid_metrics = cf.eval.confid_metrics
        self.query_monitor_plots = cf.eval.monitor_plots
        self.query_confids = cf.eval.confidence_measures

        self.output_paths = cf.exp.output_paths
        self.version_dir = cf.exp.version_dir
        self.val_every_n_epoch = cf.trainer.val_every_n_epoch
        self.running_test_train_encoded = []
        self.running_test_train_labels = []
        self.running_test_encoded = []
        self.running_test_softmax = []
        self.running_test_softmax_dist = []
        self.running_test_labels = []
        self.running_test_dataset_idx = []
        self.running_test_external_confids = []
        self.running_test_external_confids_dist = []
        self.running_confid_stats = {}
        self.running_perf_stats = {}
        self.running_confid_stats["train"] = {
            k: {"confids": [], "correct": []} for k in self.query_confids.train
        }
        self.running_confid_stats["val"] = {
            k: {"confids": [], "correct": []} for k in self.query_confids.val
        }
        self.running_val_labels = []
        self.running_train_labels = []
        self.running_train_correct_sum_sanity = 0
        self.running_val_correct_sum_sanity = 0
        self.running_perf_stats["train"] = {
            k: [] for k in self.query_performance_metrics.train
        }
        self.running_perf_stats["val"] = {
            k: [] for k in self.query_performance_metrics.val
        }

    def on_train_start(self, trainer, pl_module):
        if self.fast_dev_run is False:
            hp_metrics = {
                "hp/train_{}".format(k): 0 for k in self.query_performance_metrics
            }
            hp_metrics.update(
                {"hp/val_{}".format(k): 0 for k in self.query_performance_metrics}
            )
            pl_module.loggers[0].log_hyperparams(self.tensorboard_hparams, hp_metrics)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        loss = outputs["loss"].cpu()
        softmax = outputs["softmax"].cpu()
        y = outputs["labels"].cpu()

        tmp_correct = None
        if len(self.running_perf_stats["train"].keys()) > 0:
            stat_keys = self.running_perf_stats["train"].keys()
            y_one_hot = None
            if "loss" in stat_keys:
                self.running_perf_stats["train"]["loss"].append(loss)
            if "nll" in stat_keys:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
                self.running_perf_stats["train"]["nll"].append(
                    torch.sum(-torch.log(softmax + 1e-7) * y_one_hot, dim=1).mean()
                )
            if "accuracy" in stat_keys:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.ByteTensor)
                self.running_perf_stats["train"]["accuracy"].append(
                    tmp_correct.sum() / tmp_correct.numel()
                )
            if "brier_score" in stat_keys:
                if y_one_hot is None:
                    y_one_hot = torch.nn.functional.one_hot(
                        y, num_classes=self.num_classes
                    )
                self.running_perf_stats["train"]["brier_score"].append(
                    ((softmax - y_one_hot) ** 2).sum(1).mean()
                )

        if len(self.running_confid_stats["train"].keys()) > 0:
            self.running_train_labels.extend(y)
            self.running_train_correct_sum_sanity += tmp_correct.sum()
            stat_keys = self.running_confid_stats["train"].keys()
            if tmp_correct is None:
                tmp_correct = (torch.argmax(softmax, dim=1) == y).type(torch.ByteTensor)
            if "det_mcp" in stat_keys:
                tmp_confids = torch.max(softmax, dim=1)[0]
                self.running_confid_stats["train"]["det_mcp"]["confids"].extend(
                    tmp_confids
                )
                self.running_confid_stats["train"]["det_mcp"]["correct"].extend(
                    tmp_correct
                )
            if "det_pe" in stat_keys:
                tmp_confids = torch.sum(softmax * (-torch.log(softmax + 1e-7)), dim=1)
                self.running_confid_stats["train"]["det_pe"]["confids"].extend(
                    tmp_confids
                )
                self.running_confid_stats["train"]["det_pe"]["correct"].extend(
                    tmp_correct
                )

            if "ext" in stat_keys:
                tmp_confids = outputs["confid"].cpu()
                if tmp_confids is not None:
                    self.running_confid_stats["train"]["ext"]["confids"].extend(
                        tmp_confids
                    )
                    self.running_confid_stats["train"]["ext"]["correct"].extend(
                        tmp_correct
                    )

            if "imgs" in outputs.keys():
                eval_utils.plot_input_imgs(
                    outputs["imgs"],
                    y,
                    self.output_paths.fit.input_imgs_plot,
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            len(self.running_confid_stats["train"].keys()) > 0
            or len(self.running_perf_stats["train"].keys()) > 0
        ) and (self.running_train_correct_sum_sanity > 0):
            do_plot = (
                True
                if (pl_module.current_epoch + 1) % self.val_every_n_epoch == 0
                and len(self.query_confids.train) > 0
                and len(self.query_monitor_plots) > 0
                else False
            )
            monitor_metrics, monitor_plots = eval_utils.monitor_eval(
                self.running_confid_stats["train"],
                self.running_perf_stats["train"],
                self.running_train_labels,
                self.query_confid_metrics.train,
                self.query_monitor_plots,
                do_plot=do_plot,
                ext_confid_name=pl_module.ext_confid_name,
            )
            tqdm.write(f"CHECK TRAIN METRICS {str(monitor_metrics)}")
            tensorboard = pl_module.loggers[0].experiment
            for k, v in monitor_metrics.items():
                pl_module.log("train/{}".format(k), v, sync_dist=self.sync_dist)
                tensorboard.add_scalar(
                    "hp/train_{}".format(k), v, global_step=pl_module.current_epoch
                )

            if do_plot:
                for k, v in monitor_plots.items():
                    tensorboard.add_figure(
                        "train/{}".format(k), v, pl_module.current_epoch
                    )

        self.running_confid_stats["train"] = {
            k: {"confids": [], "correct": []} for k in self.query_confids.train
        }
        self.running_perf_stats["train"] = {
            k: [] for k in self.query_performance_metrics.train
        }
        self.running_train_correct_sum_sanity = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        tmp_correct = None
        loss = outputs["loss"]
        softmax = outputs["softmax"]
        y = outputs["labels"]
        softmax_dist = outputs.get("softmax_dist")
        perf_keys = self.running_perf_stats["val"].keys()
        confid_keys = self.running_confid_stats["val"].keys()
        if dataloader_idx is None or dataloader_idx == 0:
            self.running_val_labels.extend(outputs["labels"].cpu())
            if len(perf_keys) > 0:
                y_one_hot = None
                if "loss" in perf_keys:
                    self.running_perf_stats["val"]["loss"].append(loss)
                if "nll" in perf_keys:
                    y_one_hot = torch.nn.functional.one_hot(
                        y, num_classes=self.num_classes
                    )
                    self.running_perf_stats["val"]["nll"].append(
                        torch.sum(-torch.log(softmax + 1e-7) * y_one_hot, dim=1).mean()
                    )
                if "accuracy" in perf_keys:
                    tmp_correct = (torch.argmax(softmax, dim=1) == y).type(
                        torch.ByteTensor
                    )
                    self.running_perf_stats["val"]["accuracy"].append(
                        tmp_correct.sum() / tmp_correct.numel()
                    )
                if "brier_score" in perf_keys:
                    if y_one_hot is None:
                        y_one_hot = torch.nn.functional.one_hot(
                            y, num_classes=self.num_classes
                        )
                    self.running_perf_stats["val"]["brier_score"].append(
                        ((softmax - y_one_hot) ** 2).sum(1).mean()
                    )

            if len(confid_keys) > 0 or softmax_dist is not None:
                if tmp_correct is None:
                    tmp_correct = (torch.argmax(softmax, dim=1) == y).type(
                        torch.ByteTensor
                    )
                self.running_val_correct_sum_sanity += tmp_correct.sum()
                if "det_mcp" in confid_keys:
                    tmp_confids = torch.max(softmax, dim=1)[0]
                    self.running_confid_stats["val"]["det_mcp"]["confids"].extend(
                        tmp_confids
                    )
                    self.running_confid_stats["val"]["det_mcp"]["correct"].extend(
                        tmp_correct
                    )
                if "det_pe" in confid_keys:
                    tmp_confids = torch.sum(
                        softmax * (-torch.log(softmax + 1e-7)), dim=1
                    )
                    self.running_confid_stats["val"]["det_pe"]["confids"].extend(
                        tmp_confids
                    )
                    self.running_confid_stats["val"]["det_pe"]["correct"].extend(
                        tmp_correct
                    )

                if "ext" in confid_keys:
                    tmp_confids = outputs["confid"]
                    if tmp_confids is not None:
                        self.running_confid_stats["val"]["ext"]["confids"].extend(
                            tmp_confids
                        )
                        self.running_confid_stats["val"]["ext"]["correct"].extend(
                            tmp_correct
                        )

            if softmax_dist is not None:
                mean_softmax = torch.mean(softmax_dist, dim=2)
                tmp_mcd_correct = (torch.argmax(mean_softmax, dim=1) == y).type(
                    torch.ByteTensor
                )

                if "mcd_mcp" in confid_keys:
                    tmp_confids = torch.max(mean_softmax, dim=1)[0]
                    self.running_confid_stats["val"]["mcd_mcp"]["confids"].extend(
                        tmp_confids
                    )
                    self.running_confid_stats["val"]["mcd_mcp"]["correct"].extend(
                        tmp_mcd_correct
                    )
                if "mcd_pe" in confid_keys:
                    pe_confids = torch.sum(
                        mean_softmax * (-torch.log(mean_softmax + 1e-7)), dim=1
                    )
                    self.running_confid_stats["val"]["mcd_pe"]["confids"].extend(
                        pe_confids
                    )
                    self.running_confid_stats["val"]["mcd_pe"]["correct"].extend(
                        tmp_mcd_correct
                    )
                if "mcd_ee" in confid_keys:
                    ee_confids = torch.sum(
                        softmax_dist * (-torch.log(softmax_dist + 1e-7)), dim=1
                    ).mean(1)
                    self.running_confid_stats["val"]["mcd_ee"]["confids"].extend(
                        ee_confids
                    )
                    self.running_confid_stats["val"]["mcd_ee"]["correct"].extend(
                        tmp_mcd_correct
                    )
                if "mcd_mi" in confid_keys:
                    tmp_confids = pe_confids - ee_confids
                    self.running_confid_stats["val"]["mcd_mi"]["confids"].extend(
                        tmp_confids
                    )
                    self.running_confid_stats["val"]["mcd_mi"]["correct"].extend(
                        tmp_mcd_correct
                    )
                if "mcd_sv" in confid_keys:
                    tmp_confids = (
                        (softmax_dist - mean_softmax.unsqueeze(2)) ** 2
                    ).mean((1, 2))
                    self.running_confid_stats["val"]["mcd_sv"]["confids"].extend(
                        tmp_confids
                    )
                    self.running_confid_stats["val"]["mcd_sv"]["correct"].extend(
                        tmp_mcd_correct
                    )

        if "ood_ext" in confid_keys:
            tmp_confids = outputs["confid"]
            tmp_correct = (
                torch.zeros_like(tmp_confids)
                if dataloader_idx > 0
                else torch.ones_like(tmp_confids)
            )
            if tmp_confids is not None:
                self.running_confid_stats["val"]["ood_ext"]["confids"].extend(
                    -tmp_confids
                )
                self.running_confid_stats["val"]["ood_ext"]["correct"].extend(
                    tmp_correct
                )

        if pl_module.current_epoch == self.num_epochs - 1:
            self.running_test_softmax.extend(
                softmax_dist if softmax_dist is not None else softmax
            )
            self.running_test_labels.extend(y)
            if outputs.get("confid") is not None:
                self.running_test_external_confids.extend(outputs["confid"])

    def on_validation_epoch_end(self, trainer, pl_module):
        monitor_metrics = None
        if (
            len(self.running_confid_stats["val"].keys()) > 0
            or len(self.running_perf_stats["val"].keys()) > 0
        ) and (self.running_val_correct_sum_sanity > 0):
            do_plot = (
                True
                if len(self.query_confids.val) > 0 and len(self.query_monitor_plots) > 0
                else False
            )
            tqdm.write(
                f'{self.running_confid_stats["val"].keys()} {[len(ix["confids"]) for ix in self.running_confid_stats["val"].values()]}'
            )
            monitor_metrics, monitor_plots = eval_utils.monitor_eval(
                self.running_confid_stats["val"],
                self.running_perf_stats["val"],
                self.running_val_labels,
                self.query_confid_metrics.val,
                self.query_monitor_plots,
                do_plot=do_plot,
                ext_confid_name=pl_module.ext_confid_name,
            )
            tensorboard = pl_module.loggers[0].experiment
            for k, v in monitor_metrics.items():
                pl_module.log("val/{}".format(k), v, sync_dist=self.sync_dist)
                tensorboard.add_scalar(
                    "hp/val_{}".format(k), v, global_step=pl_module.current_epoch
                )

            if do_plot:
                for k, v in monitor_plots.items():
                    tensorboard.add_figure(
                        "val/{}".format(k), v, pl_module.current_epoch
                    )

        tqdm.write(f"CHECK VAL METRICS {str(monitor_metrics)}")
        if hasattr(pl_module, "selection_metrics"):
            for metric, mode in zip(
                pl_module.selection_metrics, pl_module.selection_modes
            ):
                if (
                    monitor_metrics is None
                    or metric.split("/")[-1] not in monitor_metrics.keys()
                ):
                    dummy = 0 if mode == "max" else 1
                    tqdm.write(
                        "selection metric {} not computed, replacing with {}.".format(
                            metric, dummy
                        )
                    )
                    pl_module.log("{}".format(metric), dummy, sync_dist=self.sync_dist)

        self.running_confid_stats["val"] = {
            k: {"confids": [], "correct": []} for k in self.query_confids.val
        }
        self.running_perf_stats["val"] = {
            k: [] for k in self.query_performance_metrics.val
        }
        self.running_val_correct_sum_sanity = 0

    def on_train_end(self, trainer, pl_module):
        # if len(self.running_test_softmax) > 0:
        if False:
            stacked_softmax = torch.stack(self.running_test_softmax, dim=0)
            stacked_labels = torch.stack(self.running_test_labels, dim=0).unsqueeze(1)
            stacked_dataset_idx = torch.zeros_like(stacked_labels)
            raw_output = torch.cat(
                [
                    stacked_softmax.reshape(stacked_softmax.size()[0], -1),
                    stacked_labels,
                    stacked_dataset_idx,
                ],
                dim=1,
            )
            np.save(self.output_paths.fit.raw_output, raw_output.cpu().data.numpy())
            tqdm.write(
                "saved raw validation outputs to {}".format(
                    self.output_paths.fit.raw_output
                )
            )

            if len(self.running_test_external_confids) > 0:
                stacked_external_confids = torch.stack(
                    self.running_test_external_confids, dim=0
                )
                np.save(
                    self.output_paths.fit.external_confids,
                    stacked_external_confids.cpu().data.numpy(),
                )
                tqdm.write(
                    "saved external confids validation outputs to {}".format(
                        self.output_paths.fit.external_confids
                    )
                )

            self.running_test_softmax = []
            self.running_test_labels = []
            self.running_test_external_confids = []

        eval_utils.clean_logging(self.version_dir)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not isinstance(outputs, dict):
            return

        if self.cfg.test.compute_train_encodings and dataloader_idx == 0:
            if outputs["encoded"] is not None:
                self.running_test_train_encoded.extend(
                    outputs["encoded"].to(dtype=torch.float16).cpu()
                )
            self.running_test_train_labels.extend(outputs["labels"].cpu())
            return

        if self.cfg.test.compute_train_encodings:
            dataloader_idx -= 1

        if outputs["encoded"] is not None:
            self.running_test_encoded.extend(
                outputs["encoded"].to(dtype=torch.float16).cpu()
            )
        self.running_test_softmax.extend(
            outputs["logits"].to(dtype=self.output_dtype).cpu()
        )
        self.running_test_labels.extend(outputs["labels"].cpu())
        if "ext" in self.query_confids.test and outputs.get("confid") is not None:
            self.running_test_external_confids.extend(outputs["confid"].cpu())
        if outputs.get("logits_dist") is not None:
            self.running_test_softmax_dist.extend(
                outputs["logits_dist"].to(dtype=self.output_dtype).cpu()
            )
        if outputs.get("confid_dist") is not None:
            self.running_test_external_confids_dist.extend(outputs["confid_dist"].cpu())

        self.running_test_dataset_idx.extend(
            torch.ones_like(outputs["labels"].cpu()) * dataloader_idx
        )

    def on_test_end(self, trainer, pl_module):
        logger.info("Saving test outputs to disk")

        stacked_softmax = torch.stack(self.running_test_softmax, dim=0)
        stacked_labels = torch.stack(self.running_test_labels, dim=0).unsqueeze(1)
        stacked_dataset_idx = torch.stack(
            self.running_test_dataset_idx, dim=0
        ).unsqueeze(1)
        raw_output = torch.cat(
            [
                stacked_softmax.reshape(stacked_softmax.size()[0], -1),
                stacked_labels,
                stacked_dataset_idx,
            ],
            dim=1,
        )
        if len(self.running_test_encoded) > 0:
            stacked_encoded = torch.stack(self.running_test_encoded, dim=0)
            encoded_output = torch.cat(
                [
                    stacked_encoded,
                    stacked_dataset_idx,
                ],
                dim=1,
            )
            np.savez_compressed(
                self.output_paths.test.encoded_output, encoded_output.cpu().data.numpy()
            )
        if len(self.running_test_train_encoded) > 0:
            stacked_train_encoded = torch.stack(self.running_test_train_encoded, dim=0)
            stacked_train_labels = torch.stack(
                self.running_test_train_labels, dim=0
            ).unsqueeze(1)
            encoded_train_output = torch.cat(
                [
                    stacked_train_encoded,
                    stacked_train_labels,
                ],
                dim=1,
            )
            np.savez_compressed(
                self.output_paths.test.encoded_train,
                encoded_train_output.cpu().data.numpy(),
            )
            w, b = pl_module.last_layer()
            w = w.cpu().numpy()
            b = b.cpu().numpy()
            np.savez_compressed(self.cfg.test.dir / "last_layer.npz", w=w, b=b)

        # try:
        #    trainer.datamodule.test_datasets[0].csv.to_csv(
        #        self.output_paths.test.attributions_output
        #    )
        try:
            for ds_idx, test_ds in enumerate(trainer.datamodule.test_datasets):
                test_ds.csv.to_csv(
                    f"{self.output_paths.test.attributions_output[:-4]}{ds_idx}.csv"
                )
        except:
            pass
        np.savez_compressed(
            self.output_paths.test.raw_output, raw_output.cpu().data.numpy()
        )
        logger.info(
            "Saved raw test outputs to {}".format(self.output_paths.test.raw_output)
        )

        if len(self.running_test_softmax_dist) > 0:
            stacked_softmax = torch.stack(self.running_test_softmax_dist, dim=0)
            np.savez_compressed(
                self.output_paths.test.raw_output_dist,
                stacked_softmax.cpu().data.numpy(),
            )
            tqdm.write(
                "saved softmax dist raw test outputs to {}".format(
                    self.output_paths.test.raw_output_dist
                )
            )

        if len(self.running_test_external_confids) > 0:
            stacked_external_confids = torch.stack(
                self.running_test_external_confids, dim=0
            )
            np.savez_compressed(
                self.output_paths.test.external_confids,
                stacked_external_confids.cpu().data.numpy(),
            )
            tqdm.write(
                "saved ext confid raw test outputs to {}".format(
                    self.output_paths.test.external_confids
                )
            )

        if len(self.running_test_external_confids_dist) > 0:
            stacked_external_confids_dist = torch.stack(
                self.running_test_external_confids_dist, dim=0
            )
            np.savez_compressed(
                self.output_paths.test.external_confids_dist,
                stacked_external_confids_dist.cpu().data.numpy(),
            )
            tqdm.write(
                "saved ext confid dist raw test outputs to {}".format(
                    self.output_paths.test.external_confids_dist
                )
            )
