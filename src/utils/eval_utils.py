import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn import metrics as skm
import seaborn
from torchmetrics import Metric
import pandas as pd

def get_tb_hparams(cf):

    hparams_collection = {
        "fold": cf.exp.fold
    }
    return {k:v for k,v in hparams_collection.items() if k in cf.eval.tb_hparams}


def monitor_eval(running_confid_stats, running_perf_stats, query_confid_metrics, query_monitor_plots, do_plot=True):

    out_metrics = {}
    out_plots = {}
    bins = 20

    # currently not implemented for mcd_softmax_mean
    for perf_key, perf_list in running_perf_stats.items():
        out_metrics[perf_key] = torch.stack(perf_list, dim=0).mean().item()

    cpu_confid_stats = {k:{} for k in list(running_confid_stats.keys())}

    for confid_key, confid_dict in running_confid_stats.items():
        confids_cpu = torch.stack(confid_dict["confids"], dim=0).cpu().data.numpy()
        correct_cpu = torch.stack(confid_dict["correct"], dim=0).cpu().data.numpy()

        if any(cfd in confid_key for cfd  in ["_pe", "_ee", "_mi", "_sv"]):
            min_confid = np.min(confids_cpu)
            max_confid = np.max(confids_cpu)
            confids_cpu = 1 - ((confids_cpu - min_confid) / (max_confid - min_confid))

        eval = ConfidEvaluator(confids=confids_cpu,
                         correct=correct_cpu,
                         query_metrics=query_confid_metrics,
                         bins=bins)

        confid_metrics = eval.get_metrics_per_confid()

        for metric_key, metric in confid_metrics.items():
            out_metrics[confid_key + "_" + metric_key] = metric
        cpu_confid_stats[confid_key]["metrics"] = confid_metrics
        cpu_confid_stats[confid_key]["plot_stats"] = eval.get_plot_stats_per_confid()
        cpu_confid_stats[confid_key]["confids"] = confids_cpu
        cpu_confid_stats[confid_key]["correct"] = correct_cpu

    if do_plot:

        plotter = ConfidPlotter(input_dict=cpu_confid_stats,
                                query_plots = query_monitor_plots,
                                bins=20,
                                performance_metrics = out_metrics)

        f = plotter.compose_plot()
        total = correct_cpu.size
        correct = np.sum(correct_cpu)
        title_string = "total: {}, corr.:{}, incorr.:{} \n".format(total,
                                                                   correct,
                                                                   total - correct
                                                                   )

        for ix, (k, v) in enumerate(out_metrics.items()):
            title_string += "{}: {:.3f} ".format(k, v)
            if (ix % 5) == 0 and ix > 0:
                title_string += "\n"
        f.suptitle(title_string)
        f.tight_layout()

        out_plots["default_plot"] = f


    return out_metrics, out_plots


class ConfidEvaluator():
    def __init__(self, confids, correct, query_metrics, bins):
        self.confids = confids
        self.correct = correct
        self.query_metrics = query_metrics
        self.bins = bins
        self.bin_accs = None
        self.bin_confids = None
        self.fpr_list = None
        self.tpr_list = None
        self.rc_curve = None
        self.precision_list = None
        self.recall_list = None

    def get_metrics_per_confid(self):

        out_metrics = {}

        if "failauc" in self.query_metrics or "fpr@95tpr" in self.query_metrics:
            if self.fpr_list is None:
                self.get_roc_curve_stats()

            if "failauc" in self.query_metrics:
                out_metrics["failauc"] = skm.auc(self.fpr_list, self.tpr_list)

            if "fpr@95tpr" in self.query_metrics:
                # soft threshold from corbiere et al. (confidnet)
                out_metrics["fpr@95tpr"] = np.min(self.fpr_list[np.argwhere(self.tpr_list >= 0.9495)])

        if "failap_suc" in self.query_metrics:
            out_metrics["failap_suc"] = skm.average_precision_score(self.correct, self.confids, pos_label=1)

        if "failap_err" in self.query_metrics:
            out_metrics["failap_err"] = skm.average_precision_score(self.correct, - self.confids, pos_label=0)

        if "aurc" in self.query_metrics or "e-aurc" in self.query_metrics:
            if self.rc_curve is None:
                self.get_rc_curve_stats()

            if "aurc" in self.query_metrics:
                out_metrics["aurc"] = self.aurc * 1000

            if "e-aurc" in self.query_metrics:
                out_metrics["e-aurc"] = self.eaurc * 1000

        hist_confids = np.histogram(self.confids, bins=self.bins, range=(0, 1))[0]
        if self.bin_accs is None:
            self.get_calibration_stats()
        bin_discrepancies = np.abs(self.bin_accs - self.bin_confids)

        if "mce" in self.query_metrics:
            out_metrics["mce"] = (bin_discrepancies).max()

        if "ece" in self.query_metrics:
            try:
                out_metrics["ece"] = \
                (np.dot(bin_discrepancies, hist_confids[np.argwhere(hist_confids > 0)]) / np.sum(hist_confids))[0]
            except:
                print("sklearn calibration failed. passing -1 for ECE.")
                out_metrics["ece"] = -1

        return out_metrics

    def get_plot_stats_per_confid(self):
        plot_stats_dict = {}

        if self.fpr_list is None:
            self.get_roc_curve_stats()
        plot_stats_dict["fpr_list"] = self.fpr_list
        plot_stats_dict["tpr_list"] = self.tpr_list

        if self.bin_accs is None:
            self.get_calibration_stats()
        plot_stats_dict["bin_accs"] = self.bin_accs
        plot_stats_dict["bin_confids"] = self.bin_confids

        if self.rc_curve is None:
            self.get_rc_curve_stats()
        plot_stats_dict["coverage_list"] = np.array(self.rc_curve[0])
        plot_stats_dict["selective_risk_list"] = np.array(self.rc_curve[1])

        if self.precision_list is None:
            self.get_err_prc_curve_stats()
        plot_stats_dict["err_precision_list"] = self.precision_list
        plot_stats_dict["err_recall_list"] =self.recall_list

        return plot_stats_dict

    def get_roc_curve_stats(self):
        self.fpr_list, self.tpr_list, _ = skm.roc_curve(self.correct, self.confids)

    def get_rc_curve_stats(self):
        self.rc_curve, self.aurc, self.eaurc = RC_curve((1 - self.correct), self.confids)

    def get_err_prc_curve_stats(self):
        self.precision_list, self.recall_list, _ = skm.precision_recall_curve(self.correct, - self.confids, pos_label=0)

    def get_calibration_stats(self):
        self.bin_accs, self.bin_confids = calibration_curve(self.correct, self.confids, n_bins=self.bins)


class ConfidPlotter():
    def __init__(self, input_dict, query_plots, bins, performance_metrics=None):
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

        self.method_names_list = []
        self.confids_list = []
        self.correct_list = []
        self.metrics_list = []
        self.colors_list = []
        self.performance_metrics = performance_metrics

        for confid_key, confid_dict in self.input_dict.items():
            self.method_names_list.append(confid_key)
            self.confids_list.append(confid_dict["confids"])
            self.metrics_list.append(confid_dict["metrics"])
            self.correct_list.append(confid_dict["correct"])

        if "hist_per_confid" in self.query_plots:
            self.query_plots = [x for x in self.query_plots if x!="hist_per_confid"]
            self.query_plots += ["{}_hist".format(x) for x in self.method_names_list]

        self.num_plots = len(self.query_plots)

    def compose_plot(self):
        seaborn.set_style('whitegrid')
        self.colors_list = seaborn.hls_palette(len(self.method_names_list)).as_hex()
        n_columns = 2
        n_rows = int(np.ceil(self.num_plots / n_columns))
        n_columns += 1
        f, axs = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(5*n_columns, 3*n_rows))
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
                method_name = ("_").join(name.split("_")[:-1])
                self.plot_hist_per_confid(method_name)

            plot_ix += 1


        legend_info = [ax.get_legend_handles_labels() for ax in f.axes]
        labels, ixs = np.unique(np.array([h for l in legend_info for h in l[1]]), return_index=True)
        handles = np.array([h for l in legend_info for h in l[0]])[ixs]
        f.legend(handles, labels, loc='upper right', prop={'size': 13})

        f.tight_layout() # this is slow af
        return f


    def plot_hist_per_confid(self, method_name):

        confids = self.confids_list[self.method_names_list.index(method_name)]
        correct = self.correct_list[self.method_names_list.index(method_name)]

        self.ax.hist(confids[np.argwhere(correct == 1)],
                      color="g",
                      bins=self.bins,
                      width=1 / self.bins,
                      alpha=0.3,
                      label="correct")

        self.ax.hist(confids[np.argwhere(correct == 0)],
                      color="r",
                      bins=self.bins,
                      width=1 / self.bins,
                      alpha=0.3,
                      label="incorrect")

        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_yscale('log')
        self.ax.set_xlabel("Confid")
        title_string = method_name
        if self.performance_metrics is not None:
            title_string += " (acc: {:.3f}, total:{})".format(self.performance_metrics["accuracy"], confids.size)
        self.ax.set_title("failure_pred_{}".format(title_string))


    def plot_calibration(self):

        bin_confids_list = []
        bin_accs_list = []
        for confid_key, confid_dict in self.input_dict.items():
            bin_confids_list.append(confid_dict["plot_stats"]["bin_confids"])
            bin_accs_list.append(confid_dict["plot_stats"]["bin_accs"])

        for name, bin_confid, bin_acc, color, metrics in zip(self.method_names_list,
                                                             bin_confids_list,
                                                             bin_accs_list,
                                                             self.colors_list,
                                                             self.metrics_list):
            label = name
            if "ece" in metrics.keys():
                label += " (ece: {:.3f})".format(metrics["ece"])
            self.ax.plot(bin_confid, bin_acc, marker="o", markersize=3, color=color, label=label)
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

        for name, bin_confid, bin_acc, color, metrics in zip(self.method_names_list,
                                                    bin_confids_list,
                                                    bin_accs_list,
                                                    self.colors_list,
                                                    self.metrics_list):
            label = name
            if "ece" in metrics.keys():
                label += " (ece: {:.3f})".format(metrics["ece"])
            self.ax.plot(bin_confid, bin_confid - bin_acc, marker="o", markersize=3, label=label, color=color)
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

        for name, fpr, tpr, color, metrics in zip(self.method_names_list,
                                                    fpr_list,
                                                    tpr_list,
                                                    self.colors_list,
                                                    self.metrics_list):
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

        for name, precision, recall, color, metrics in zip(self.method_names_list,
                                                    precision_list,
                                                    recall_list,
                                                    self.colors_list,
                                                      self.metrics_list):
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

        for confid_key, confid_dict in self.input_dict.items():
            coverage_list.append(confid_dict["plot_stats"]["coverage_list"])
            selective_risk_list.append(confid_dict["plot_stats"]["selective_risk_list"])

        for name, coverage, selective_risk, color, metrics in zip(self.method_names_list,
                                                    coverage_list,
                                                    selective_risk_list,
                                                    self.colors_list,
                                                    self.metrics_list):
            label = name
            if "aurc" in metrics.keys():
                label += " (aurc%: {:.3f})".format(metrics["aurc"]*100)
            self.ax.plot(coverage, selective_risk * 100, label=label, color=color)
        self.ax.set_title("RC Curve")
        self.ax.set_ylabel("Selective Risk [%]")
        self.ax.set_xlabel("Coverage")


def RC_curve(residuals, confidence):
    # from https://github.com/geifmany/uncertainty_ICLR/blob/495f82d9d9a24e1dd62e62dd1f86d78e4f53a471/utils/uncertainty_tools.py#L13
    # residuals = inverted "correct_list"
    # implemented for risk = 0/1 error.
    # could be changed to other error (e.g NLL?, that would weirdly mix up kappa confidence with predictive uncertainty!)
    # curve = []
    # n = len(residuals)
    # idx_sorted = np.argsort(confidence)
    # temp1 = residuals[idx_sorted]
    # cov = n
    # selective_risk = sum(temp1)
    # curve.append((cov/ n, selective_risk / n))
    # for i in range(0, len(idx_sorted)-1):
    #     cov = cov-1
    #     selective_risk = selective_risk-residuals[idx_sorted[i]]
    #     # if confidence[idx_sorted[i]] != confidence[idx_sorted[np.max(i - 1, 0)]]:
    #     curve.append((cov / n, selective_risk /(n-i - 1)))      # Todo: I Correceted this. report!!
    # AUC = sum([a[1] for a in curve])/len(curve)
    # err = np.mean(residuals)
    # kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    # EAURC = AUC-kappa_star_aurc
    # print("MY RC", AUC, EAURC, curve[-10:])
    # print("MY AURC WITH SKM", skm.auc([c[0] for c in curve], [c[1] for c in curve]))
    # print("MY EAURC WITH SKM", skm.auc([c[0] for c in curve], [c[1] for c in curve]))

    # version from corbeire et al.
    accuracy = 1-np.mean(residuals)
    proba_pred = confidence
    accurate = 1-residuals
    risks, coverages = [], []
    for delta in sorted(set(proba_pred))[:-1]:
        coverages.append((proba_pred > delta).mean())
        selected_accurate = accurate[proba_pred > delta]
        risks.append(1. - selected_accurate.mean())
    aurc = skm.auc(coverages, risks)
    eaurc = aurc - ((1. - accuracy) + accuracy * np.log(accuracy))
    curve = (coverages, risks)

    return curve, aurc, eaurc

 #
 # risks, coverages = [], []
 #    for delta in sorted(set(self.proba_pred))[:-1]:
 #        coverages.append((self.proba_pred > delta).mean())
 #        selected_accurate = self.accurate[self.proba_pred > delta]
 #        risks.append(1. - selected_accurate.mean())
 #    aurc = auc(coverages, risks)
 #    eaurc = aurc - ((1. - accuracy) + accuracy*np.log(accuracy))




class BrierScore(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.add_state("brier_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
  #      preds, target = self._input_format(preds, target)


        y_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        assert preds.shape == y_one_hot.shape

        self.brier_score += ((preds - y_one_hot) ** 2).sum(1).mean()
        self.total += 1

    def compute(self):
        # compute final result
        return self.brier_score.float() / self.total


def clean_logging(log_dir):
    df = pd.read_csv(os.path.join(log_dir, "metrics.csv"))
    df = df.groupby("step").max()
    df.to_csv(os.path.join(log_dir, "metrics.csv"))

