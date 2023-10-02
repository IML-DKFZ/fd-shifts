import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn
import torch
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve
from torchmetrics import Metric

from . import logger
from .metrics import StatsCache, get_metric_function


def _get_tb_hparams(cf):
    hparams_collection = {"fold": cf.exp.fold}
    return {k: v for k, v in hparams_collection.items() if k in cf.eval.tb_hparams}


def monitor_eval(
    running_confid_stats,
    running_perf_stats,
    running_labels,
    query_confid_metrics,
    query_monitor_plots,
    do_plot=True,
    ext_confid_name=None,
):
    out_metrics = {}
    out_plots = {}
    bins = 20
    labels_cpu = torch.stack(running_labels, dim=0).cpu().data.numpy()

    # currently not implemented for mcd_softmax_mean
    for perf_key, perf_list in running_perf_stats.items():
        out_metrics[perf_key] = torch.stack(perf_list, dim=0).mean().item()

    cpu_confid_stats = {}

    for confid_key, confid_dict in running_confid_stats.items():
        if len(confid_dict["confids"]) > 0:
            confids_cpu = torch.stack(confid_dict["confids"], dim=0).cpu().data.numpy()
            correct_cpu = torch.stack(confid_dict["correct"], dim=0).cpu().data.numpy()

            if confid_key == "ext" and ext_confid_name == "bpd":
                out_metrics["bpd_mean"] = np.mean(confids_cpu)

            if any(cfd in confid_key for cfd in ["_pe", "_ee", "_mi", "_sv"]) or (
                confid_key == "ext" and ext_confid_name == "bpd"
            ):
                min_confid = np.min(confids_cpu)
                max_confid = np.max(confids_cpu)
                confids_cpu = 1 - (
                    (confids_cpu - min_confid) / (max_confid - min_confid + 1e-9)
                )

            if confid_key == "ext" and ext_confid_name == "maha":
                confids_cpu = (confids_cpu - confids_cpu.min()) / np.abs(
                    confids_cpu.min() - confids_cpu.max()
                )

            if confid_key == "ood_ext":
                query_confid_metrics = ["failauc"]
                query_monitor_plots = ["hist_per_confid"]

            eval = ConfidEvaluator(
                confids=confids_cpu,
                correct=correct_cpu,
                labels=labels_cpu,
                query_metrics=query_confid_metrics,
                query_plots=query_monitor_plots,
                bins=bins,
            )
            confid_metrics = eval.get_metrics_per_confid()

            for metric_key, metric in confid_metrics.items():
                out_metrics[confid_key + "_" + metric_key] = metric

            cpu_confid_stats[confid_key] = {}
            cpu_confid_stats[confid_key]["metrics"] = confid_metrics
            cpu_confid_stats[confid_key][
                "plot_stats"
            ] = eval.get_plot_stats_per_confid()
            cpu_confid_stats[confid_key]["confids"] = confids_cpu
            cpu_confid_stats[confid_key]["correct"] = correct_cpu

    if do_plot and len(cpu_confid_stats) > 0:
        plotter = ConfidPlotter(
            input_dict=cpu_confid_stats,
            query_plots=query_monitor_plots,
            bins=20,
            performance_metrics=out_metrics,
        )

        f = plotter.compose_plot()
        total = correct_cpu.size
        correct = np.sum(correct_cpu)
        title_string = "total: {}, corr.:{}, incorr.:{} \n".format(
            total, correct, total - correct
        )

        for ix, (k, v) in enumerate(out_metrics.items()):
            title_string += "{}: {:.3f} ".format(k, v)
            if (ix % 5) == 0 and ix > 0:
                title_string += "\n"
        f.suptitle(title_string)
        f.tight_layout()

        out_plots["default_plot"] = f

    return out_metrics, out_plots


class ConfidEvaluator:
    def __init__(self, confids, correct, labels, query_metrics, query_plots, bins):
        self.confids = confids[~np.isnan(confids)]
        self.correct = correct[~np.isnan(confids)]
        self.query_metrics = query_metrics
        self.query_plots = query_plots
        self.bins = bins
        self.bin_accs = None
        self.bin_confids = None
        self.fpr_list = None
        self.tpr_list = None
        self.rc_curve = None
        self.precision_list = None
        self.recall_list = None
        self.labels = labels
        self.stats_cache = StatsCache(
            self.confids, self.correct, self.bins, self.labels
        )

    def get_metrics_per_confid(self):
        out_metrics = {}
        if "failauc" in self.query_metrics or "fpr@95tpr" in self.query_metrics:
            if "failauc" in self.query_metrics:
                out_metrics["failauc"] = get_metric_function("failauc")(
                    self.stats_cache
                )
            if "fpr@95tpr" in self.query_metrics:
                out_metrics["fpr@95tpr"] = get_metric_function("fpr@95tpr")(
                    self.stats_cache
                )

        if "failap_suc" in self.query_metrics:
            out_metrics["failap_suc"] = get_metric_function("failap_suc")(
                self.stats_cache
            )
        if "failap_err" in self.query_metrics:
            out_metrics["failap_err"] = get_metric_function("failap_err")(
                self.stats_cache
            )

        if (
            "aurc" in self.query_metrics
            or "e-aurc" in self.query_metrics
            or "b-aurc" in self.query_metrics
        ):
            if self.rc_curve is None:
                self.get_rc_curve_stats()
            if "aurc" in self.query_metrics:
                out_metrics["aurc"] = get_metric_function("aurc")(self.stats_cache)
            if "b-aurc" in self.query_metrics:
                out_metrics["b-aurc"] = get_metric_function("b-aurc")(self.stats_cache)
            if "e-aurc" in self.query_metrics:
                out_metrics["e-aurc"] = get_metric_function("e-aurc")(self.stats_cache)

            if "risk@95cov" in self.query_metrics:
                coverages = np.array(self.rc_curve[0])
                risks = np.array(self.rc_curve[1])
                out_metrics["risk@100cov"] = (
                    np.min(risks[np.argwhere(coverages >= 1)]) * 100
                )
                out_metrics["risk@95cov"] = (
                    np.min(risks[np.argwhere(coverages >= 0.95)]) * 100
                )
                out_metrics["risk@90cov"] = (
                    np.min(risks[np.argwhere(coverages >= 0.90)]) * 100
                )
                out_metrics["risk@85cov"] = (
                    np.min(risks[np.argwhere(coverages >= 0.85)]) * 100
                )
                out_metrics["risk@80cov"] = (
                    np.min(risks[np.argwhere(coverages >= 0.80)]) * 100
                )
                out_metrics["risk@75cov"] = (
                    np.min(risks[np.argwhere(coverages >= 0.75)]) * 100
                )

        if self.bin_accs is None:
            self.get_calibration_stats()

        if "mce" in self.query_metrics:
            out_metrics["mce"] = get_metric_function("mce")(self.stats_cache)

        if "ece" in self.query_metrics:
            out_metrics["ece"] = get_metric_function("ece")(self.stats_cache)

        if "fail-NLL" in self.query_metrics:
            out_metrics["fail-NLL"] = get_metric_function("fail-NLL")(self.stats_cache)
            logger.debug(
                "CHECK FAIL NLL: \n{}\n{}", self.confids.max(), self.confids.min()
            )

        return out_metrics

    def get_plot_stats_per_confid(self):
        plot_stats_dict = {}

        if "roc_curve" in self.query_plots:
            if self.fpr_list is None:
                self.get_roc_curve_stats()
            plot_stats_dict["fpr_list"] = self.fpr_list
            plot_stats_dict["tpr_list"] = self.tpr_list

        if "calibration" in self.query_plots or "overconfidence" in self.query_plots:
            if self.bin_accs is None:
                self.get_calibration_stats()
            plot_stats_dict["bin_accs"] = self.bin_accs
            plot_stats_dict["bin_confids"] = self.bin_confids

        if "rc_curve" in self.query_plots:
            if self.rc_curve is None:
                self.get_rc_curve_stats()
            plot_stats_dict["coverage_list"] = np.array(self.rc_curve[0])
            plot_stats_dict["selective_risk_list"] = np.array(self.rc_curve[1])

        if "prc_curve" in self.query_plots:
            if self.precision_list is None:
                self.get_err_prc_curve_stats()
            plot_stats_dict["err_precision_list"] = self.precision_list
            plot_stats_dict["err_recall_list"] = self.recall_list

        return plot_stats_dict

    def get_roc_curve_stats(self):
        try:
            self.fpr_list, self.tpr_list, _ = skm.roc_curve(self.correct, self.confids)
        except:
            logger.debug(
                "FAIL CHECK\n{}\n{}\n{}\n{}\n{}\n{}",
                self.correct.shape,
                self.confids.shape,
                np.min(self.correct),
                np.max(self.correct),
                np.min(self.confids),
                np.max(self.confids),
            )

    def get_rc_curve_stats(self):
        self.rc_curve, self.aurc, self.eaurc = RC_curve(
            (1 - self.correct), self.confids
        )

    def get_err_prc_curve_stats(self):
        self.precision_list, self.recall_list, _ = skm.precision_recall_curve(
            self.correct, -self.confids, pos_label=0
        )

    def get_calibration_stats(self):
        calib_confids = np.clip(self.confids, 0, 1)
        self.bin_accs, self.bin_confids = calibration_curve(
            self.correct, calib_confids, n_bins=self.bins
        )

    def calculate_bound(self, delta, m, erm):
        # This function is a solver for the inverse of binomial CDF based on binary search.
        precision = 1e-9

        def func(b):
            return (-1 * delta) + scipy.stats.binom.cdf(int(m * erm), m, b)

        a = erm  # start binary search from the empirical risk
        c = 1  # the upper bound is 1
        b = (a + c) / 2  # mid point
        funcval = func(b)
        while abs(funcval) > precision:
            if a == 1.0 and c == 1.0:
                b = 1.0
                break
            elif funcval > 0:
                a = b
            else:
                c = b
            b = (a + c) / 2
            funcval = func(b)
        return b

    def get_val_risk_scores(self, rstar, delta, no_bound_mode=False):
        """A function to calculate the risk bound proposed in the paper, the algorithm is based on algorithm 1 from the paper.

        Args:
            rstar (): the requested risk bound
            delta (): the desired delta
            kappa (): rating function over the points (higher values is more confident prediction)
            residuals (): a vector of the residuals of the samples 0 is correct prediction and 1 corresponding to an error
            split (): is a boolean controls whether to split train and test

        Returns:
            [theta, bound] (also prints latex text for the tables in the paper)
        """

        val_risk_scores = {}
        probs = self.confids
        FY = 1 - self.correct

        m = len(FY)

        probs_idx_sorted = np.argsort(probs)

        a = 0
        b = m - 1
        deltahat = delta / math.ceil(math.log2(m))

        for q in range(math.ceil(math.log2(m)) + 1):
            mid = math.ceil((a + b) / 2)

            mi = len(FY[probs_idx_sorted[mid:]])
            theta = probs[probs_idx_sorted[mid]]
            risk = sum(FY[probs_idx_sorted[mid:]]) / mi
            bound = self.calculate_bound(deltahat, mi, risk)
            coverage = mi / m
            if no_bound_mode:
                bound = risk
            if bound > rstar:
                a = mid
            else:
                b = mid

        val_risk_scores["val_risk"] = risk
        val_risk_scores["val_cov"] = coverage
        val_risk_scores["theta"] = theta
        logger.debug(
            "STRAIGHT FROM THRESH CALCULATION\n{}\n{}\n{}\n{}\n{}\n{}",
            risk,
            coverage,
            theta,
            rstar,
            delta,
            bound,
        )
        return val_risk_scores


class ConfidPlotter:
    def __init__(
        self, input_dict, query_plots, bins, performance_metrics=None, fig_scale=1
    ):
        """
        input list ist a list of methods dicts, each with keys:
        cfg, exp, correct, confid_types
        confid_types is a dict with keys
        mcp, pe, ...
        each of which is a confid dict again with keys
        plot_stats, confids , ...
        NEW: gets input dict of confid dicts! each one is one method to compare!
        """
        self.input_dict = input_dict
        self.query_plots = query_plots
        self.bins = bins
        self.ax = None

        self.confid_keys_list = []
        self.confids_list = []
        self.correct_list = []
        self.metrics_list = []
        self.colors_list = []
        self.performance_metrics = performance_metrics
        self.fig_scale = fig_scale

        for confid_key, confid_dict in self.input_dict.items():
            self.confid_keys_list.append(confid_key)
            self.confids_list.append(confid_dict["confids"])
            self.metrics_list.append(confid_dict["metrics"])
            self.correct_list.append(confid_dict["correct"])

        if "hist_per_confid" in self.query_plots:
            self.query_plots = [x for x in self.query_plots if x != "hist_per_confid"]
            self.query_plots += ["{}_hist".format(x) for x in self.confid_keys_list]

        self.num_plots = len(self.query_plots)
        self.threshold = None

    def compose_plot(self):
        seaborn.set(font_scale=self.fig_scale, style="whitegrid")
        self.colors_list = seaborn.hls_palette(len(self.confid_keys_list)).as_hex()
        n_columns = 2
        n_rows = int(np.ceil(self.num_plots / n_columns))
        n_columns += 1
        f, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(5 * n_columns * self.fig_scale, 3 * n_rows * self.fig_scale),
        )
        plot_ix = 0
        for ix in range(len(f.axes)):
            if (ix + 1) % n_columns == 0 or plot_ix >= len(self.query_plots):
                f.axes[ix].axis("off")
                continue

            self.ax = f.axes[ix]
            name = self.query_plots[plot_ix]
            if name == "calibration":
                self.plot_calibration()
            if name == "overconfidence":
                self.plot_overconfidence()
            if name == "roc_curve":
                self.plot_roc()
            if name == "prc_curve":
                self.plot_prc()
            if name == "rc_curve":
                self.plot_rc()
            if "_hist" in name:
                confid_key = ("_").join(name.split("_")[:-1])
                self.plot_hist_per_confid(confid_key)

            plot_ix += 1

        legend_info = [ax.get_legend_handles_labels() for ax in f.axes]
        labels, ixs = np.unique(
            np.array([h for l in legend_info for h in l[1]]), return_index=True
        )
        handles = np.array([h for l in legend_info for h in l[0]])[ixs]
        f.legend(handles, labels, loc="upper right", prop={"size": 10 * self.fig_scale})

        f.tight_layout()
        return f

    def plot_hist_per_confid(self, confid_key):
        min_plot_x = 0
        confids = self.confids_list[self.confid_keys_list.index(confid_key)]
        correct = self.correct_list[self.confid_keys_list.index(confid_key)]

        if confid_key == "ood_ext" or self.threshold is not None:
            custom_range = (np.min(confids), np.max(confids))
            (n_correct, binsc, patchesc) = self.ax.hist(
                confids[np.argwhere(correct == 1)],
                color="g",
                bins=self.bins,
                range=custom_range,
                alpha=0.3,
                label="correct",
            )

            (n_incorrect, bins, patches) = self.ax.hist(
                confids[np.argwhere(correct == 0)],
                color="r",
                bins=self.bins,
                range=custom_range,
                alpha=0.3,
                label="incorrect",
            )
        else:
            (n_correct, binsc, patchesc) = self.ax.hist(
                confids[np.argwhere(correct == 1)],
                color="g",
                bins=self.bins,
                range=(min_plot_x, 1),
                width=0.9 * (1 - min_plot_x) / (self.bins),
                alpha=0.3,
                label="correct",
            )

            (n_incorrect, bins, patches) = self.ax.hist(
                confids[np.argwhere(correct == 0)],
                color="r",
                bins=self.bins,
                range=(min_plot_x, 1),
                width=0.9 * (1 - min_plot_x) / (self.bins),
                alpha=0.3,
                label="incorrect",
            )

        max_y_data = np.max([np.max(n_correct), np.max(n_incorrect)])
        if not confid_key == "ood_ext":
            self.ax.set_xlim(min_plot_x - 0.1, 1.1)
            self.ax.set_ylim(0.1, max_y_data)
        self.ax.vlines(
            np.mean(confids[np.argwhere(correct == 0)]),
            ymin=0,
            ymax=max_y_data,
            color="r",
            linestyles="-",
            label="incorrect mean",
        )
        self.ax.vlines(
            np.median(confids[np.argwhere(correct == 0)]),
            ymin=0,
            ymax=max_y_data,
            color="r",
            linestyles="--",
            label="incorrect median",
        )
        self.ax.vlines(
            np.mean(confids[np.argwhere(correct == 1)]),
            ymin=0,
            ymax=max_y_data,
            color="g",
            linestyles="-",
            label="correct mean",
        )
        self.ax.vlines(
            np.median(confids[np.argwhere(correct == 1)]),
            ymin=0,
            ymax=max_y_data,
            color="g",
            linestyles="--",
            label="correct median",
        )
        if self.threshold is not None:
            self.ax.vlines(
                self.threshold,
                ymin=0,
                ymax=max_y_data,
                color="b",
                linestyles="--",
                label="risk threshold",
                linewidth=4,
            )

        self.ax.set_yscale("log")
        self.ax.set_xlabel("Confid")
        title_string = confid_key
        title_string += " (incorr.:{}, tot:{})".format(
            correct.size - correct.sum(), correct.size
        )
        self.ax.set_title("{}".format(title_string))

    def plot_calibration(self):
        bin_confids_list = []
        bin_accs_list = []
        for confid_key, confid_dict in self.input_dict.items():
            bin_confids_list.append(confid_dict["plot_stats"]["bin_confids"])
            bin_accs_list.append(confid_dict["plot_stats"]["bin_accs"])

        for name, bin_confid, bin_acc, color, metrics in zip(
            self.confid_keys_list,
            bin_confids_list,
            bin_accs_list,
            self.colors_list,
            self.metrics_list,
        ):
            label = name
            if "ece" in metrics.keys():
                label += " (ece: {:.3f})".format(metrics["ece"])
            self.ax.plot(
                bin_confid, bin_acc, marker="o", markersize=3, color=color, label=label
            )
        self.ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
        self.ax.set_ylabel("Acc")
        self.ax.set_xlabel("Confid")
        self.ax.set_title("calibration")

    def plot_overconfidence(self):
        bin_confids_list = []
        bin_accs_list = []
        for confid_key, confid_dict in self.input_dict.items():
            bin_confids_list.append(confid_dict["plot_stats"]["bin_confids"])
            bin_accs_list.append(confid_dict["plot_stats"]["bin_accs"])

        for name, bin_confid, bin_acc, color, metrics in zip(
            self.confid_keys_list,
            bin_confids_list,
            bin_accs_list,
            self.colors_list,
            self.metrics_list,
        ):
            label = name
            if "ece" in metrics.keys():
                label += " (ece: {:.3f})".format(metrics["ece"])
            self.ax.plot(
                bin_confid,
                bin_confid - bin_acc,
                marker="o",
                markersize=3,
                label=label,
                color=color,
            )
        self.ax.plot([0, 1], [0, 0], linestyle="--", color="black", alpha=0.5)
        self.ax.set_title("overconfidence")
        self.ax.set_ylabel("Confid - Acc")
        self.ax.set_xlabel("Confid")

    def plot_roc(self):
        fpr_list = []
        tpr_list = []

        for confid_key, confid_dict in self.input_dict.items():
            fpr_list.append(confid_dict["plot_stats"]["fpr_list"])
            tpr_list.append(confid_dict["plot_stats"]["tpr_list"])

        for name, fpr, tpr, color, metrics in zip(
            self.confid_keys_list,
            fpr_list,
            tpr_list,
            self.colors_list,
            self.metrics_list,
        ):
            label = name
            if "failauc" in metrics.keys():
                label += " (auc: {:.3f})".format(metrics["failauc"])
            self.ax.plot(fpr, tpr, label=label, color=color)
        self.ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
        self.ax.set_title("ROC Curve")
        self.ax.set_ylabel("TPR")
        self.ax.set_xlabel("FPR")

    def plot_prc(self):
        precision_list = []
        recall_list = []

        for confid_key, confid_dict in self.input_dict.items():
            precision_list.append(confid_dict["plot_stats"]["err_precision_list"])
            recall_list.append(confid_dict["plot_stats"]["err_recall_list"])

        for name, precision, recall, color, metrics in zip(
            self.confid_keys_list,
            precision_list,
            recall_list,
            self.colors_list,
            self.metrics_list,
        ):
            label = name
            if "failap_err" in metrics.keys():
                label += " (ap_err: {:.3f})".format(metrics["failap_err"])
            self.ax.plot(recall, precision, label=label, color=color)
        self.ax.set_title("PRC Curve (Error=Positive)")
        self.ax.set_ylabel("Precision")
        self.ax.set_xlabel("Recall")

    def plot_rc(self):
        coverage_list = []
        selective_risk_list = []
        coverage_list_geif = []
        selective_risk_list_geif = []

        for confid_key, confid_dict in self.input_dict.items():
            coverage_list.append(confid_dict["plot_stats"]["coverage_list"])
            selective_risk_list.append(confid_dict["plot_stats"]["selective_risk_list"])

        for name, coverage, selective_risk, color, metrics in zip(
            self.confid_keys_list,
            coverage_list,
            selective_risk_list,
            self.colors_list,
            self.metrics_list,
        ):
            label = name
            if "aurc" in metrics.keys():
                label += " (aurc%: {:.3f})".format(metrics["aurc"] * 100)
            self.ax.plot(
                coverage, selective_risk * 100, label=label, color=color, alpha=1
            )
        self.ax.set_title("RC Curve")
        self.ax.set_ylabel("Selective Risk [%]")
        self.ax.set_xlabel("Coverage")


def RC_curve(residuals, confidence):
    coverages = []
    risks = []
    n = len(residuals)
    idx_sorted = np.argsort(confidence)
    cov = n
    error_sum = sum(residuals[idx_sorted])
    coverages.append(cov / n),
    risks.append(error_sum / n)
    weights = []
    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        cov = cov - 1
        error_sum = error_sum - residuals[idx_sorted[i]]
        selective_risk = error_sum / (n - 1 - i)
        tmp_weight += 1
        if i == 0 or confidence[idx_sorted[i]] != confidence[idx_sorted[i - 1]]:
            coverages.append(cov / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    # add a well-defined final point to the RC-curve.
    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / n)

    # aurc is computed as a weighted average over risk scores analogously to the average precision score.
    aurc = sum([a * w for a, w in zip(risks, weights)])

    # compute e-aurc
    err = np.mean(residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    e_aurc = aurc - kappa_star_aurc

    curve = (coverages, risks)
    return curve, aurc, e_aurc


class BrierScore(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.add_state("brier_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states

        y_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        assert preds.shape == y_one_hot.shape

        self.brier_score += ((preds - y_one_hot) ** 2).sum(1).mean()
        self.total += 1

    def compute(self):
        # compute final result
        return self.brier_score.float() / self.total


def clean_logging(log_dir):
    try:
        df = pd.read_csv(log_dir / "metrics.csv")
        df = df.groupby("step").max().round(3)
        df.to_csv(log_dir / "metrics.csv")
    except:
        logger.warning("no metrics.csv found in clean logging!")


def plot_input_imgs(x, y, out_path):
    logger.debug(
        "{}\n{}\n{}\n{}",
        x.mean().item(),
        x.std().item(),
        x.min().item(),
        x.max().item(),
    )
    f, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    for ix in range(len(f.axes)):
        ax = f.axes[ix]
        ax.imshow(x[ix].cpu().permute(1, 2, 0))
        ax.title.set_text(str(y[ix].item()))

    plt.tight_layout()
    plt.savefig(out_path)
    assert 1 == 2


def qual_plot(fp_dict, fn_dict, out_path):
    n_rows = len(fp_dict["images"])
    f, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(6, 13))
    title_pad = 0.85
    fontsize = 22

    col = 0  # FP
    for d in [fp_dict, fn_dict]:
        if len(d["images"]) > 0:
            for row in range(n_rows):
                label = d["labels"][row]
                if isinstance(label, str) and len(label) > 9:
                    label = label[:10] + "."
                predict = d["predicts"][row]
                if isinstance(predict, str) and len(predict) > 9:
                    predict = predict[:10] + "."
                ax = axs[row, col]
                ax.imshow(d["images"][row].permute(1, 2, 0))
                titel_string = ""
                titel_string += "true: {} \n".format(label)
                titel_string += "pred.: {} \n".format(predict)
                titel_string += "confid.: {:.3f} \n".format(d["confids"][row])
                ax.set_title(titel_string, loc="left", fontsize=fontsize, y=title_pad)
                ax.axis("off")

        col += 1

    plt.subplots_adjust(wspace=0.23, hspace=0.4)
    f.savefig(out_path)
    plt.close()
    logger.debug("saved qual_plot to {}", out_path)


def ThresholdPlot(plot_dict):
    scale = 10
    n_cols = len(plot_dict)
    n_rows = 1
    colors = ["b", "k", "purple"]
    f, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(n_cols * scale * 0.6, n_rows * scale * 0.4)
    )

    logger.debug("plot in {}", len(plot_dict))
    for ix, (study, study_dict) in enumerate(plot_dict.items()):
        logger.debug("threshold plot {} {}", study, len(study_dict["confids"]))
        confids = study_dict["confids"]
        correct = study_dict["correct"]
        delta_threshs = study_dict["delta_threshs"]
        plot_string = study_dict["plot_string"]
        deltas = study_dict["deltas"]
        true_thresh = study_dict["true_thresh"]

        custom_range = (np.min(confids), np.max(confids))
        (n_correct, binsc, patchesc) = axs[ix].hist(
            confids[np.argwhere(correct == 1)],
            color="g",
            bins=20,
            range=custom_range,
            alpha=0.3,
            label="correct",
        )

        (n_incorrect, bins, patches) = axs[ix].hist(
            confids[np.argwhere(correct == 0)],
            color="r",
            bins=20,
            range=custom_range,
            alpha=0.3,
            label="incorrect",
        )

        for idx, dt in enumerate(delta_threshs):
            logger.debug("drawing line", idx, dt, delta_threshs, deltas)
            axs[ix].vlines(
                dt,
                ymin=0,
                ymax=axs[ix].get_ylim()[1],
                label="thresh_delta_{}".format(deltas[idx]),
                linestyles="-",
                linewidth=2.5,
                color=colors[idx],
            )
        axs[ix].vlines(
            true_thresh,
            ymin=0,
            ymax=axs[ix].get_ylim()[1],
            label="thresh_r*",
            linestyles="-",
            linewidth=3,
            color="greenyellow",
        )

        axs[ix].set_yscale("log")
        axs[ix].set_xlabel(study)
        axs[ix].set_title(plot_string)

    plt.legend()
    plt.tight_layout()
    return f


cifar100_classes = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]
