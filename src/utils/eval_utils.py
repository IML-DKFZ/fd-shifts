import torch
from torch.nn import functional as F
import os
from PIL import Image
from torch.distributions.normal import Normal
from torchvision import transforms
import math
import random
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


def monitor_eval(confids_dict, correct, query_monitor_metrics, query_monitor_plots, do_plot=True):

    out_metrics = {}
    out_plots = {}
    bins = 20
    correct_cpu = torch.stack(correct, dim=0).cpu().data.numpy()

    method_dict = {"name": "", "correct": correct_cpu}
    method_dict["confid_types"] = {k:{} for k in list(confids_dict.keys())}

    for confid_key, confids in confids_dict.items():

        confids_cpu = torch.stack(confids, dim=0).cpu().data.numpy()

        if confid_key == "pe":
            min_confid = np.min(confids_cpu)
            max_confid = np.max(confids_cpu)
            confids_cpu = 1 - ((confids_cpu - min_confid) / (max_confid - min_confid))

        eval = ConfidEvaluator(confids=confids_cpu,
                         correct=correct_cpu,
                         query_metrics=query_monitor_metrics,
                         bins=bins)

        confid_metrics = eval.get_metrics_per_confid()

        for metric_key, metric in confid_metrics.items():
            out_metrics[confid_key + "_" + metric_key] = metric

        method_dict["confid_types"][confid_key]["metrics"] = confid_metrics
        method_dict["confid_types"][confid_key]["plot_stats"] = eval.get_plot_stats_per_confid()
        method_dict["confid_types"][confid_key]["confids"] = confids_cpu

    if len(query_monitor_plots)>0 and do_plot==True:

        plotter = ConfidPlotter(input_list=[method_dict],
                                query_plots = query_monitor_plots,
                                bins=20)

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

        if "accuracy" in self.query_metrics:
            out_metrics["accuracy"] = np.sum(self.correct) / self.correct.size

        if "failauc" in self.query_metrics or "fpr@95tpr" in self.query_metrics:
            if self.fpr_list is None:
                self.get_roc_curve_stats()

            if "failauc" in self.query_metrics:
                out_metrics["failauc"] = skm.auc(self.fpr_list, self.tpr_list)

            if "fpr@95tpr" in self.query_metrics:
                out_metrics["fpr@95tpr"] = np.min(self.fpr_list[np.argwhere(self.tpr_list >= 0.95)])

        if "failap_suc" in self.query_metrics:
            out_metrics["failap_suc"] = skm.average_precision_score(self.correct, self.confids, pos_label=1)

        if "failap_err" in self.query_metrics:
            out_metrics["failap_err"] = skm.average_precision_score(self.correct, - self.confids, pos_label=0)

        if "aurc" in self.query_metrics or "e-aurc" in self.query_metrics:
            if self.rc_curve is None:
                self.get_rc_curve_stats()

            if "aurc" in self.query_metrics:
                out_metrics["aurc"] = self.aurc

            if "e-aurc" in self.query_metrics:
                out_metrics["e-aurc"] = self.eaurc

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
        plot_stats_dict["coverage_list"] = np.array([x[0] for x in self.rc_curve])
        plot_stats_dict["selective_risk_list"] = np.array([x[1] for x in self.rc_curve])

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
    def __init__(self, input_list, query_plots, bins):
        """
        input list ist a list of methods dicts, each with keys:
        cfg, exp, correct, confid_types
        confid_types is a dict with keys
        mcp, pe, ...
        each of which is a confid dict again with keys
        plot_stats, confids , ...
        """
        self.input_list = input_list
        self.query_plots = query_plots
        self.bins = bins
        self.ax = None

        self.method_names_list = []
        self.confids_list = []
        self.correct_list = []
        self.metrics_list = []
        self.colors_list = ["g", "b", "r", "y"]

        for method_dict in self.input_list:
            for confid_key, confid_dict in method_dict["confid_types"].items():
                name = "{}_{}".format(method_dict["name"], confid_key) if len(method_dict["name"])>0 else confid_key
                self.method_names_list.append(name)
                self.confids_list.append(confid_dict["confids"])
                self.metrics_list.append(confid_dict["metrics"])
                self.correct_list.append(method_dict["correct"])

        if "hist_per_confid" in self.query_plots:
            self.query_plots = [x for x in self.query_plots if x!="hist_per_confid"]
            self.query_plots += ["{}_hist".format(x) for x in self.method_names_list]

        self.num_plots = len(self.query_plots)

    def compose_plot(self):
        seaborn.set_style('whitegrid')
        n_columns = 2
        n_rows = int(np.ceil(self.num_plots / n_columns))
        f, _ = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(5*n_columns, 3*n_rows))

        for name, ax in zip(self.query_plots, f.axes):
            self.ax = ax
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

        f.tight_layout() # this is slow af
        return f


    def plot_hist_per_confid(self, method_name):

        confids = self.confids_list[self.method_names_list.index(method_name)]
        correct = self.correct_list[self.method_names_list.index(method_name)]
        metrics = self.metrics_list[self.method_names_list.index(method_name)]

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
        self.ax.legend(loc=1)
        title_string = method_name
        if metrics is not None:
            title_string += " (acc: {:.3f}, total:{})".format(metrics["accuracy"], confids.size)
        self.ax.set_title("failure_pred_{}".format(title_string))


    def plot_calibration(self):

        bin_confids_list = []
        bin_accs_list = []
        for method_dict in self.input_list:
            for confid_key, confid_dict in method_dict["confid_types"].items():
                bin_confids_list.append(confid_dict["plot_stats"]["bin_confids"])
                bin_accs_list.append(confid_dict["plot_stats"]["bin_accs"])

        for name, bin_confid, bin_acc, color, metrics in zip(self.method_names_list,
                                                             bin_confids_list,
                                                             bin_accs_list,
                                                             self.colors_list,
                                                             self.metrics_list):

            label = name
            if metrics is not None:
                label += " (ece: {:.3f})".format(metrics["ece"])
            self.ax.plot(bin_confid, bin_acc, marker="o", label=label)
        self.ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
        self.ax.legend(loc="lower right")
        self.ax.set_ylabel("Acc")
        self.ax.set_xlabel("Confid")
        self.ax.set_title("calibration")

    def plot_overconfidence(self):

        bin_confids_list = []
        bin_accs_list = []
        for method_dict in self.input_list:
            for confid_key, confid_dict in method_dict["confid_types"].items():
                bin_confids_list.append(confid_dict["plot_stats"]["bin_confids"])
                bin_accs_list.append(confid_dict["plot_stats"]["bin_accs"])

        for name, bin_confid, bin_acc, color, metric in zip(self.method_names_list,
                                                    bin_confids_list,
                                                    bin_accs_list,
                                                    self.colors_list,
                                                    self.metrics_list):
            label = name
            if metric is not None:
                label += " (ece: {:.3f})".format(metric["ece"])
            self.ax.plot(bin_confid, bin_confid - bin_acc, marker="o",label=label)
        self.ax.plot([0, 1], [0, 0], linestyle="--", color="black", alpha=0.5)
        self.ax.legend(loc="lower right")
        self.ax.set_title("overconfidence")
        self.ax.set_ylabel("Confid - Acc")
        self.ax.set_xlabel("Confid")


    def plot_roc(self):

        fpr_list = []
        tpr_list = []

        for method_dict in self.input_list:
            for confid_key, confid_dict in method_dict["confid_types"].items():
                fpr_list.append(confid_dict["plot_stats"]["fpr_list"])
                tpr_list.append(confid_dict["plot_stats"]["tpr_list"])

        for name, fpr, tpr, color, metric in zip(self.method_names_list,
                                                    fpr_list,
                                                    tpr_list,
                                                    self.colors_list,
                                                    self.metrics_list):
            label = name
            if metric is not None:
                label += " (auc: {:.3f})".format(metric["failauc"])
            self.ax.plot(fpr, tpr, label=label)
        self.ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
        self.ax.legend(loc="lower right")
        self.ax.set_title("ROC Curve")
        self.ax.set_ylabel("TPR")
        self.ax.set_xlabel("FPR")

    def plot_prc(self):

        precision_list = []
        recall_list = []

        for method_dict in self.input_list:
            for confid_key, confid_dict in method_dict["confid_types"].items():
                precision_list.append(confid_dict["plot_stats"]["err_precision_list"])
                recall_list.append(confid_dict["plot_stats"]["err_recall_list"])

        for name, precision, recall, color, metric in zip(self.method_names_list,
                                                    precision_list,
                                                    recall_list,
                                                    self.colors_list,
                                                      self.metrics_list):
            label = name
            if metric is not None:
                label += " (ap_err: {:.3f})".format(metric["failap_err"])
            self.ax.plot(recall, precision, label=label)
        self.ax.legend(loc="lower right")
        self.ax.set_title("PRC Curve (Error=Positive)")
        self.ax.set_ylabel("Precision")
        self.ax.set_xlabel("Recall")

    def plot_rc(self):

        coverage_list = []
        selective_risk_list = []

        for method_dict in self.input_list:
            for confid_key, confid_dict in method_dict["confid_types"].items():
                coverage_list.append(confid_dict["plot_stats"]["coverage_list"])
                selective_risk_list.append(confid_dict["plot_stats"]["selective_risk_list"])

        for name, coverage, selective_risk, color, metric in zip(self.method_names_list,
                                                    coverage_list,
                                                    selective_risk_list,
                                                    self.colors_list,
                                                    self.metrics_list):
            label = name
            if metric is not None:
                label += " (aurc%: {:.3f})".format(metric["aurc"]*100)
            self.ax.plot(coverage, selective_risk * 100, label=label)
        self.ax.legend(loc="upper left")
        self.ax.set_title("RC Curve")
        self.ax.set_ylabel("Selective Risk [%]")
        self.ax.set_xlabel("Coverage")


def RC_curve(residuals, confidence):
    # from https://github.com/geifmany/uncertainty_ICLR/blob/495f82d9d9a24e1dd62e62dd1f86d78e4f53a471/utils/uncertainty_tools.py#L13
    # residuals = inverted "correct_list"
    # implemented for risk = 0/1 error.
    # could be changed to other error (e.g NLL?, that would weirdly mix up kappa confidence with predictive uncertainty!)
    curve = []
    n = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = n
    selective_risk = sum(temp1)
    curve.append((cov/ n, selective_risk / n))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        selective_risk = selective_risk-residuals[idx_sorted[i]]
        curve.append((cov / n, selective_risk /(n-i)))
    AUC = sum([a[1] for a in curve])/len(curve)
    err = np.mean(residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    EAURC = AUC-kappa_star_aurc
    return curve, AUC, EAURC



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



def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    # from albumentation examples
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def append_metric_dict(metric_dict, metric_str, results_dict, plot_dict):

    res_per_concept = []
    n_unspecified_res = {k:[] for k in range(len(list(metric_dict.keys())) + 1)}

    for k, v in metric_dict.items():
        mean_cat = np.mean(v['value']) if len(v['value']) > 0 else None
        results_dict['write_out']['{}_{}'.format(metric_str, k)] = mean_cat
        res_per_concept.extend(v['value'])
        for kix, vv in enumerate(v['value']):
            n_unspecified_res[v['n_unspecified'][kix]].append(vv)
    results_dict['write_out']['{}_{}'.format(metric_str, 'total')] = np.mean(res_per_concept) if len(res_per_concept) > 0 else np.nan
    plot_dict['{}_{}'.format(metric_str, 'total')] = res_per_concept

    for k, v in n_unspecified_res.items():
        plot_dict['{}_{}_{}'.format(metric_str, k, 'unsp.')] = v
        results_dict['write_out']['{}_{}_{}'.format(metric_str, k, 'unsp.')] = np.mean(v) if len(v) > 0 else None

    return results_dict, plot_dict


def get_factor_vae_metric(args, list_of_all_representations, list_of_reps_for_factorvae_metric, modality='y'):

    # list_of_all_representations: list([number_samples, z_dim], [], [], ....)
    # list_of_reps_for_factorvae_metric: list([number_samples, z_dim], [], [], ....)

    cat_all_reps = torch.cat(list_of_all_representations, 0)
    print('cat all reps size', cat_all_reps.size())
    global_variances = np.var(cat_all_reps.view(-1, cat_all_reps.size()[2]).cpu().data.numpy(), axis=0, ddof=1)
    global_means = np.mean(cat_all_reps.view(-1, cat_all_reps.size()[2]).cpu().data.numpy(), axis=0)
    active_dims = ((abs(global_variances - 1) > 0.1) | (abs(global_means - 0) > 0.1))
    print("ACTIVE DIMS {}".format(modality), np.sum(active_dims))
    global_argmins = np.argsort(global_variances[active_dims])[::-1][:3]
    accuracies = {}

    for split in list_of_reps_for_factorvae_metric.keys():
        split_of_reps_for_factorvae_metric = list_of_reps_for_factorvae_metric[split]
        votes = np.zeros((len(args.attribute_dict), global_variances[active_dims].shape[0]), dtype=np.int64)
        all_variance_ratios = []
        argmin_variance_ratios = []
        list_of_plot_info = {'circles_list':[], 'points_list':[]}
        for reps_y, k, k_value, consistencies, preds, q_y_loc, q_y_scale, agg_x_post, reps_x in split_of_reps_for_factorvae_metric:
            reps = reps_y if modality == 'y' else reps_x
            local_variances = np.var(reps.cpu().data.numpy(), axis=0, ddof=1)
            local_means = np.mean(reps.cpu().data.numpy(), axis=0)
            variance_ratios = local_variances[active_dims] / global_variances[active_dims]
            argmin = np.argmin(variance_ratios)

            print(split,
                  k,
                  k_value,
                  argmin,
                  np.round(variance_ratios,3),
                  np.round(local_means[active_dims],3),
                  np.round(q_y_loc.cpu().data.numpy()[active_dims], 3),
                  np.round(q_y_scale.cpu().data.numpy()[active_dims], 3))

            if k_value == ['cylinder', 'gray'] and split == 'all_inworld':
                correct_mean = np.mean([reps[ix].cpu().data.numpy()[active_dims]  for ix in range(reps.shape[0]) if consistencies[ix].item() == 1], axis=0)
                false_mean = np.mean([reps[ix].cpu().data.numpy()[active_dims]   for ix in range(reps.shape[0]) if consistencies[ix].item() == 0], axis=0)
                correct_var = np.var([reps[ix].cpu().data.numpy()[active_dims] for ix in range(reps.shape[0]) if consistencies[ix].item() == 1], axis=0)# / global_variances[active_dims]
                false_var = np.var([reps[ix].cpu().data.numpy()[active_dims] for ix in range(reps.shape[0]) if consistencies[ix].item() == 0], axis=0) #/ global_variances[active_dims]
                print('correct mean', np.round(correct_mean, 3))
                print('false mean', np.round(false_mean, 3))
                print('correct var', np.round(correct_var, 3))
                print('false var', np.round(false_var, 3))

                for ix in range(30):
                    p = []
                    for pix in range(2):
                        p.append(preds[pix][ix].item())
                    print(np.round(reps[ix].cpu().data.numpy()[active_dims], 3), consistencies[ix].item(), p)

            if k!= - 1:
                votes[k.item(), argmin] += 1
                all_variance_ratios.append(variance_ratios)
                argmin_variance_ratios.append(variance_ratios[argmin])

            if k == -1 and modality == 'y':
                argmins = np.argsort(variance_ratios)[:3]
                center = q_y_loc.cpu().data.numpy()[active_dims][argmins]
                radius = q_y_scale.cpu().data.numpy()[active_dims][argmins]
                list_of_plot_info['circles_list'].append([center, radius, k_value, 'y'])

                if split == 'all_inworld':
                    center = agg_x_post.loc.cpu().data.numpy()[active_dims][argmins]
                    radius = agg_x_post.scale.cpu().data.numpy()[active_dims][argmins]
                    list_of_plot_info['circles_list'].append([center, radius, k_value, 'x'])


                if 'cylinder' in k_value:
                    for ix in range(100):
                        center = reps[ix].cpu().data.numpy()[active_dims][argmins]
                        list_of_plot_info['points_list'].append([center, k_value, consistencies[ix].item()])

        gt_votes = np.argmax(votes, axis=1)
        attr_accs = votes[np.arange(votes.shape[0]), gt_votes] / votes.sum(axis=1)
        all_variance_ratios = np.array(all_variance_ratios)
        argmin_variance_ratios = np.array(argmin_variance_ratios)
        distance_to_real_min = abs(argmin_variance_ratios - all_variance_ratios[np.arange(argmin_variance_ratios.shape[0]), gt_votes[np.array([i[1] for i in split_of_reps_for_factorvae_metric if i[1]!=-1])]])
        for kix, attr_key in enumerate(args.attribute_dict.keys()):
          accuracies['fvae_{}_{}_{}'.format(modality, split, attr_key)] = attr_accs[kix]
        accuracies['fvae_{}_{}_{}'.format(modality, split, 'err_dist')] = np.mean(distance_to_real_min)

    print(accuracies)
    return accuracies, list_of_plot_info


def get_js_div(p, q):
    return 0.5 * get_kl(p, (p+q)*0.5) + 0.5 * get_kl(q, (p+q)*0.5)

def get_kl(p,q):
    return (p * ((p + 1e-6).div(q + 1e-6)).log2()).sum()



def swap_experiments(args, logger, net, ec):

    random_crop = transforms.RandomCrop(size=(args.img_size))
    to_tensor = transforms.ToTensor()

    swap_concepts = [
        [['sphere', 'blue'], ['cylinder', 'blue']],
        [['sphere', 'yellow'], ['sphere', 'red']],
        [['sphere', 'green'], ['cylinder', 'purple']],
        [['sphere', 'cyan'], ['cylinder', 'cyan']],
        [['cylinder', 'purple'], ['cylinder', 'cyan']],
       [['sphere'], ['cylinder']],
    ]

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    n_samples = 10
    n_rows = 3 * len(swap_concepts)
    n_columns = n_samples * 4
    f = plt.figure(figsize=(n_columns * 2, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, n_columns)
    gs.update(wspace=0.1, hspace=0.2)
    rix, cix = 0, 0

    for sc in swap_concepts:
        sc1 = [ix for ix in args.test_concepts if ix['string'] == sc[0]][0]
        sc2 = [ix for ix in args.test_concepts if ix['string'] == sc[1]][0]
        in_y = torch.FloatTensor([sc1[args.attribute_encode], sc2[args.attribute_encode]]).cuda()
        if (args.missing_attr_mode != 'poe' or args.model == 'mpriors') and args.attribute_encode == 'khot':
            in_y[in_y == -1] = args.fill_missing_value
        in_y -= 0.5
        img1 = to_tensor(random_crop(Image.open(os.path.join(args.data_dir, "images", sc1['example_imgs'][1] + '.png')).convert('RGB')))
        img2 = to_tensor(random_crop(Image.open(os.path.join(args.data_dir, "images", sc2['example_imgs'][0] + '.png')).convert('RGB')))
        imgs_cat = torch.cat([img1.unsqueeze(0), img2.unsqueeze(0)]).cuda()
        in_x = (imgs_cat - (args.dataset_mean / 255)) / (args.dataset_std / 255)

        mu_x, sigma_x = net.encoder_x(in_x)
        [mu_y, sigma_y], det_q_y = net.encoder_y(in_y)
        q_x = Normal(loc=mu_x, scale=sigma_x)
        q_y = Normal(loc=mu_y, scale=sigma_y)
        z_x = torch.cat([q_x.sample() for _ in range(n_samples)]) #+ 100
        z_y = torch.cat([q_y.sample() for _ in range(n_samples)])
        det_q_y = det_q_y.repeat(n_samples, 1)

        in_x_dec_list = [net.merge_modalities(z_x[0::2], det_q_y[0::2]),
                         net.merge_modalities(z_x[0::2], det_q_y[1::2]),
                         net.merge_modalities(z_x[1::2], det_q_y[0::2]),
                         net.merge_modalities(z_x[1::2], det_q_y[1::2]),
                         net.merge_modalities(z_y[0::2], det_q_y[0::2]),
                         net.merge_modalities(z_y[0::2], det_q_y[1::2]),
                         net.merge_modalities(z_y[1::2], det_q_y[0::2]),
                         net.merge_modalities(z_y[1::2], det_q_y[1::2]),
                         ]
        in_y_dec_list = [z_x, z_y]

        out_y_x_1 = utils.map_binary_attributes_to_text(args, model_utils.map_y_output(args, net.decoder_y(z_x))[0])
        out_y_y_1 = utils.map_binary_attributes_to_text(args, model_utils.map_y_output(args, net.decoder_y(z_y))[0])
        out_y_x_2 = utils.map_binary_attributes_to_text(args, model_utils.map_y_output(args, net.decoder_y(z_x))[1])
        out_y_y_2 = utils.map_binary_attributes_to_text(args, model_utils.map_y_output(args, net.decoder_y(z_y))[1])

        entropy_prior = (sigma_y * 2 * math.pi * math.e).mul(1).mean().log()*0.5
        entropy_post = (sigma_x * 2 * math.pi * math.e).mul(1).mean().log()*0.5

        for xix, x in enumerate(imgs_cat):
            ax = plt.subplot(gs[cix, rix])
            ax.imshow(x.cpu().permute(1, 2, 0).data.numpy())
            ax.axis('off')
            rix += 1
            print(rix, cix, 'in_x')
        cix += 1
        rix = 0
        print(rix, cix, 'after_in_x')

        for six, series in enumerate(in_x_dec_list):
            out_x = net.decoder_x(series)[0]

            for xix, x in enumerate(out_x):
                ax = plt.subplot(gs[cix, rix])
                ax.imshow(x.cpu().permute(1, 2, 0).data.numpy())
                if rix == n_columns//2 and cix%2==0:
                    ax.set_title('in_y: c1:{} c2:{} yx_out: {} {} yy_out:{} {} entropy: {}pr {}po'.format(sc[0], sc[1], out_y_x_1, out_y_x_2, out_y_y_1, out_y_y_2, entropy_prior, entropy_post))
                else:
                  #  ax.set_title(str(six))
                    pass
                rix +=1
                if rix == n_samples * 4:
                    rix = 0
                    cix += 1
                ax.axis('off')

    out_dir = os.path.join(args.results_root_path, args.exp_group, 'results_per_job')
    out_plot_dir = os.path.join(args.results_root_path, args.exp_group, 'plots', args.experiment_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_plot_dir):
        os.makedirs(out_plot_dir)
    if hasattr(args, 'current_epoch'):
        current_epoch = args.current_epoch
    else:
        current_epoch = 'test'
    plt.savefig(os.path.join(out_plot_dir, '{}_swapexps_{}.png').format(args.experiment_name, current_epoch))
    print('plotting to {}'.format(os.path.join(out_plot_dir, '{}_swapexps_{}.png').format(args.experiment_name, current_epoch)))
    #assert 1==2
    plt.close()


def mpriors_swap_experiments(args, logger, net, ec):

    random_crop = transforms.RandomCrop(size=(args.img_size))
    to_tensor = transforms.ToTensor()

    # ((color == 'green' & shape == 'cylinder') |  (size == 'medium' & shape == 'sphere') | (color == 'green' & material == 'rubber') | (size == 'medium' & color == 'green') | (material == 'metal' & size == 'large')
    if 'complex' in args.attribute_info_pickle_name:
        swap_concepts = [
            [['sphere', 'blue', 'metal', 'large'], ['cylinder', 'blue', 'metal', 'large']],
            [['sphere', 'yellow', 'rubber', 'medium'], ['sphere', 'red', 'rubber', 'medium']],
            [['sphere', 'cyan', 'rubber'], ['sphere', 'metal', 'small']],
            [['cylinder', 'rubber'], ['cube', 'metal']],
            [['sphere', 'large'], ['sphere', 'medium']],
            [['sphere', 'green', 'metal', 'small'], ['sphere', 'green', 'metal', 'large']],
            [['cylinder', 'green', 'metal', 'large'], ['cylinder', 'green', 'rubber', 'medium']],
            [['cube', 'green'], ['sphere', 'green']],
            [['sphere'], ['cylinder']],
        ]
    else:
        swap_concepts = [[['sphere', 'blue'], ['cylinder', 'blue']],
                         [['sphere', 'yellow'], ['sphere', 'red']],
                         [['sphere', 'cyan'], ['cylinder', 'green']],
                         [['sphere', 'cyan'], ['cylinder', 'cyan']],
                         [['sphere'], ['cylinder']]]

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    n_samples = 5
    n_rows = 3 * len(swap_concepts)
    n_columns = n_samples * 4
    f = plt.figure(figsize=(n_columns * 2, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, n_columns)
    gs.update(wspace=0.1, hspace=0.2)
    rix, cix = 0, 0

    for sc in swap_concepts:
        sc1 = [ix for ix in args.test_concepts if ix['string'] == sc[0]][0]
        sc2 = [ix for ix in args.test_concepts if ix['string'] == sc[1]][0]
        in_y = torch.FloatTensor([sc1[args.attribute_encode], sc2[args.attribute_encode]]).cuda()
        in_y -= 0.5
        img1 = to_tensor(random_crop(Image.open(os.path.join(args.data_dir, "images", sc1['example_imgs'][1] + '.png')).convert('RGB')))
        img2 = to_tensor(random_crop(Image.open(os.path.join(args.data_dir, "images", sc2['example_imgs'][0] + '.png')).convert('RGB')))
        imgs_cat = torch.cat([img1.unsqueeze(0), img2.unsqueeze(0)]).cuda()
        in_x = (imgs_cat - (args.dataset_mean / 255)) / (args.dataset_std / 255)

        mu_x, sigma_x = net.encoder_x(in_x)
        out_y = net.encoder_y(in_y)
        q_x = Normal(loc=mu_x, scale=sigma_x)
        p_x = Normal(loc=torch.zeros([in_y.shape[0], args.z_dim]).cuda(),
                         scale=torch.ones([in_y.shape[0], args.z_dim]).cuda())
        p_y = Normal(loc=torch.zeros([in_y.shape[0], args.z_dim]).cuda(),
                         scale=torch.ones([in_y.shape[0], args.z_dim]).cuda())
        z_x = torch.cat([q_x.sample() for _ in range(n_samples)]) #+ 100
        z_x_prior = torch.cat([p_x.sample() for _ in range(n_samples)]) #+ 100
        z_y_list = []
        # for prior in out_y:
        #     q_y = Normal(loc=prior[0], scale=prior[1])
        #     z_y.append(torch.cat([q_y.sample() for _ in range(n_samples)]))
        # z_y = torch.cat(z_y, 1)
        attr_ix_begin = 0
        if args.model == 'mpriors':
            for ix, det_prior in enumerate(out_y):
                attr_ix_end = attr_ix_begin + len(args.attribute_dict[list(args.attribute_dict.keys())[ix]])
                in_y_attr = (in_y[:, attr_ix_begin: attr_ix_end]).min(1)[0].repeat(n_samples)
                attr_prior = det_prior[0].clone().repeat(n_samples, 1)
                sampled_prior = torch.cat([p_y.sample() for _ in range(n_samples)])# select mu
                unsepcified_batch_elements = (in_y_attr == -1.5)
                attr_prior[unsepcified_batch_elements] = sampled_prior[unsepcified_batch_elements]
                z_y_list.append(attr_prior)
                attr_ix_begin = attr_ix_end
            z_y = torch.cat(z_y_list, 1)

        if args.model == 'deterministiccvae' and args.missing_attr_mode == 'poe':
            [mu, sigma], _ = out_y
            q_y = Normal(loc=mu, scale=sigma)
            z_y = torch.cat([q_y.sample() for _ in range(n_samples)])

        if args.model == 'deterministiccvae' and (args.data_sample_mode):
            z_y = torch.cat([in_y for _ in range(n_samples)])

        elif args.model == 'deterministiccvae' and  float(args.loss_weights_dict[0]['kl_x'][-1]) > 0:
            [mu, sigma], det_y = out_y
            z_y = torch.cat([det_y for _ in range(n_samples)])

        if args.kill_det_info:
            z_y *= 0

        y_info = torch.cat([in_y for _ in range(n_samples)]) if args.append_y_maps else None


        in_x_dec_list = [[torch.cat((z_x[0::2], z_y[0::2]), 1), y_info[0::2]],
                         [torch.cat((z_x[0::2], z_y[1::2]), 1), y_info[1::2]],
                         [torch.cat((z_x[1::2], z_y[0::2]), 1), y_info[0::2]],
                         [torch.cat((z_x[1::2], z_y[1::2]), 1), y_info[1::2]],
                         [torch.cat((z_x_prior[0::2], z_y[0::2]), 1), y_info[0::2]],
                         [torch.cat((z_x_prior[0::2], z_y[1::2]), 1), y_info[1::2]],
                         [torch.cat((z_x_prior[1::2], z_y[0::2]), 1), y_info[0::2]],
                         [torch.cat((z_x_prior[1::2], z_y[1::2]), 1), y_info[1::2]]
                         ]

        f.suptitle('unseen: {}' .format(args.data_split_queries['unseen_inworld']), fontsize=15)
        for xix, x in enumerate(imgs_cat):
            ax = plt.subplot(gs[cix, rix])
            ax.imshow(x.cpu().permute(1, 2, 0).data.numpy())
            ax.axis('off')
            rix += 1
        cix += 1
        rix = 0

        for six, series in enumerate(in_x_dec_list):
            out_x = net.decoder_x(series[0], series[1])[0]

            for xix, x in enumerate(out_x):
                ax = plt.subplot(gs[cix, rix])
                ax.imshow(x.cpu().permute(1, 2, 0).data.numpy())
                if rix == n_columns // 2 and cix % 3 == 1:
                    ax.set_title(
                        'in_y: c:{}'.format(sc))
                rix +=1
                if rix == n_samples * 4:
                    rix = 0
                    cix += 1
                ax.axis('off')

    out_dir = os.path.join(args.results_root_path, args.exp_group, 'results_per_job')
    out_plot_dir = os.path.join(args.results_root_path, args.exp_group, 'plots', args.experiment_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_plot_dir):
        os.makedirs(out_plot_dir)
    if hasattr(args, 'current_epoch'):
        current_epoch = args.current_epoch
    else:
        current_epoch = 'test'
    plt.savefig(os.path.join(out_plot_dir, '{}_mprior_swaps_{}.png').format(args.experiment_name, current_epoch))
    print('plotting to {}'.format(os.path.join(out_plot_dir, '{}_mpriors_swaps_{}.png').format(args.experiment_name, current_epoch)))
    #assert 1==2
    plt.close()



def stoch_visualization(args, logger, net, ec):

    random_crop = transforms.RandomCrop(size=(args.img_size))
    to_tensor = transforms.ToTensor()

    swap_concepts = [['sphere', 'blue'], ['sphere', 'yellow'], ['cylinder', 'purple'], ['cylinder', 'cyan'], ['red'], ['sphere']]

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    n_samples = 10
    n_rows = 3 * len(swap_concepts)
    n_columns = n_samples
    f = plt.figure(figsize=(n_columns * 2, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, n_columns)
    gs.update(wspace=0.1, hspace=0.2)
    rix, cix = 0, 0

    for sc in swap_concepts:
        sc1 = [ix for ix in args.test_concepts if ix['string'] == sc][0]
        in_y = torch.FloatTensor([sc1[args.attribute_encode]]).cuda()
        if args.missing_attr_mode != 'poe' and args.attribute_encode == 'khot':
            in_y[in_y == -1] = args.fill_missing_value
        in_y -= 0.5
        img1 = to_tensor(random_crop(Image.open(os.path.join(args.data_dir, "images", sc1['example_imgs'][1] + '.png')).convert('RGB')))
        imgs_cat = torch.cat([img1.unsqueeze(0)]).cuda()
        in_x = (imgs_cat - (args.dataset_mean / 255)) / (args.dataset_std / 255)

        mu_x, sigma_x = net.encoder_x(in_x)
        mu_y, sigma_y = net.encoder_y(in_y)
        q_x = Normal(loc=mu_x, scale=sigma_x)
        q_y = Normal(loc=mu_y, scale=sigma_y)
        z_x = torch.cat([q_x.sample() for _ in range(n_samples)])  # + 100
        z_y = torch.cat([q_y.sample() for _ in range(n_samples)])

        in_x_dec_list = [z_x, z_y]

        out_y_x_1 = utils.map_binary_attributes_to_text(args, model_utils.map_y_output(args, net.decoder_y(z_x))[0])
        out_y_y_1 = utils.map_binary_attributes_to_text(args, model_utils.map_y_output(args, net.decoder_y(z_y))[0])

        entropy_prior = (sigma_y * 2 * math.pi * math.e).mul(1).mean().log() * 0.5
        entropy_post = (sigma_x * 2 * math.pi * math.e).mul(1).mean().log() * 0.5

        for xix, x in enumerate(imgs_cat):
            ax = plt.subplot(gs[cix, rix])
            ax.imshow(x.cpu().permute(1, 2, 0).data.numpy())
            ax.axis('off')
            rix += 1
            print(rix, cix, 'in_x')
        cix += 1
        rix = 0
        print(rix, cix, 'after_in_x')

        for six, series in enumerate(in_x_dec_list):
            out_x = net.decoder_x(series)[0]

            for xix, x in enumerate(out_x):
                ax = plt.subplot(gs[cix, rix])
                ax.imshow(x.cpu().permute(1, 2, 0).data.numpy())
                if rix == n_columns // 2 and cix % 2 == 0:
                    ax.set_title(
                        'in_y: c:{} yx_out: {} yy_out:{} entropy: {}pr {}po'.format(sc, out_y_x_1,
                                                                                        out_y_y_1,
                                                                                        entropy_prior,
                                                                                        entropy_post))

                else:
                    #  ax.set_title(str(six))
                    pass
                rix += 1
                if rix == n_samples:
                    rix = 0
                    cix += 1
                ax.axis('off')


    out_dir = os.path.join(args.results_root_path, args.exp_group, 'results_per_job')
    out_plot_dir = os.path.join(args.results_root_path, args.exp_group, 'plots', args.experiment_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_plot_dir):
        os.makedirs(out_plot_dir)
    if hasattr(args, 'current_epoch'):
        current_epoch = args.current_epoch
    else:
        current_epoch = 'test'
    plt.savefig(os.path.join(out_plot_dir, '{}_stoch_vis_{}.png').format(args.experiment_name, current_epoch))
    print('plotting to {}'.format(os.path.join(out_plot_dir, '{}_stoch_vis_{}.png').format(args.experiment_name, current_epoch)))
    plt.close()
    #assert 1==2


def test_sym2img(args, logger, net=None):

    import pandas as pd
    file_df = pd.read_csv(os.path.join(args.data_dir, args.info_df_name))
    train_df = file_df.query(args.data_split_queries['train']).reset_index()
    args.train_marginals = {}
    for k, v in args.attribute_dict.items():
        args.train_marginals[k] = []
        for av in v:
            print(k, av, v)
            args.train_marginals[k].append(len(train_df[train_df[k] == av]) / len(train_df))
    print(args.train_marginals)
    args.test_batch_size = 200
    args.n_test_samples_per_concept = 300

    if net is None:
        net = utils.import_module('model', args.model_path).net(args, logger).cuda()
        logger.info('loading test params from {} for model {}'.format(args.checkpoint_path, args.model))
        utils.load_test_params(args, args.checkpoint_path, net)

    ec = utils.import_module('ec', os.path.join(args.exec_dir, 'models', args.eval_classifier_model)).net(args, logger).cuda()
    logger.info('loading eval params from {}'.format(args.eval_classifier_params_path))
    utils.load_test_params(args, args.eval_classifier_params_path, ec, load_eval=True)



    consistency_cl = {k: {'value': [], 'n_unspecified': []} for k in list(args.data_split_queries.keys())}
    consistency_al = {k: {'value': [], 'n_unspecified': []} for k in list(args.data_split_queries.keys())}
    confusion_matrix = {k: {'pred': [], 'gt': []} for k in list(args.attribute_dict.keys())}
    coverage = {k: {'value': [], 'n_unspecified': []} for k in list(args.data_split_queries.keys())}
    comb_metric = {k: {'value': [], 'n_unspecified': []} for k in list(args.data_split_queries.keys())}
    frequency = {k: {'value': [], 'n_unspecified': []} for k in list(args.data_split_queries.keys())}
    total_variation_distance = {k: {'value': [], 'n_unspecified': []} for k in list(args.data_split_queries.keys())}
    y_reps = []
    x_reps = []
    list_of_reps_for_factorvae_metric = {k: [] for k in list(args.data_split_queries.keys())}
    random_crop = transforms.RandomCrop(size=(args.img_size))
    to_tensor = transforms.ToTensor()

    test_concepts_list = args.test_concepts
    if (hasattr(args, 'current_epoch') and args.current_epoch < args.num_epochs) and len(test_concepts_list) > 600:
        random.shuffle(test_concepts_list)
        test_concepts_list = test_concepts_list[:600]

    with torch.no_grad():

        net.eval()
        ec.eval()

        # if args.model == 'deterministiccvae' and args.xinfo_mode == 'none' and args.missing_attr_mode !='poe' and float(args.loss_weights_dict[0]['kl_xy'][-1]) > 0:
        #     swap_experiments(args, logger, net, ec)
        # elif args.model == 'mpriors' or (args.model == 'deterministiccvae' and args.missing_attr_mode=='poe')\
        #         or args.model == 'deterministiccvae':
        #     mpriors_swap_experiments(args, logger, net, ec)
        # else:
        #     pass
        #     stoch_visualization(args, logger, net, ec)

        in_y = np.array([c[args.attribute_encode] for c in test_concepts_list])
        if args.missing_attr_mode != 'poe' and args.attribute_encode == 'khot' and args.model != 'mpriors' and not args.data_sample_mode:
            in_y[in_y == -1] = args.fill_missing_value

        logger.info('starting testing over {} concepts and {} samples'.format(len(in_y), args.n_test_samples_per_concept))

        # different gpu load during testing (the number of test_concepts and n_sampled_images_per_concept can
        # be significantly higher than batch sizes during training requires a different test_batch_size. Additional gpu load
        # is required to save some images for output monitoring.
        # The list of test concepts will be processed in chunks of test_batch_size.
        split_ixs_concepts = np.split(np.arange(len(in_y)), np.arange(len(in_y))[::args.test_batch_size])
        for chunk_ixs_concepts in split_ixs_concepts[1:]:
            chunk_imgs = []
            chunk_y_out = []
            chunk_y_z = []

            net.test_encode(torch.FloatTensor(in_y[chunk_ixs_concepts] -0.5 ).cuda())

            # loop over the number of images to be sampled. reconstruct the image and append it to chunk_imgs.
            for s in range(args.n_test_samples_per_concept):
                out_x, out_y, z_y = net.test_sample_and_reco()
                chunk_imgs.append(out_x.unsqueeze(1))
                chunk_y_out.append(model_utils.map_y_output(args, out_y).unsqueeze(1))
                chunk_y_z.append(z_y.unsqueeze(1))

            imgs = torch.cat(chunk_imgs, 1)
            y_out = torch.cat(chunk_y_out, 1)
            z_y = torch.cat(chunk_y_z, 1)
            y_reps.append(z_y)

            # loop over the current chunk of test concepts again, and forward all n_sampeld_images for the current concept in batches through the eval_classifier.
            for cix in range(len(chunk_ixs_concepts)):

                current_concept = test_concepts_list[chunk_ixs_concepts[cix]]

                # save some pre-selected concepts/images for monitoring.
                if current_concept['plot_flag']:
                    current_concept['imgs'] = imgs[cix][:args.show_n_test_samples_per_concept].cpu().data.numpy()
                    current_concept['y_out'] = y_out[cix][:args.show_n_test_samples_per_concept].cpu().data.numpy()

                # process the n_sampeld_images in chunks of test_batch_size.
                sample_chunk_out_list = []
                split_ixs_samples = np.split(np.arange(args.n_test_samples_per_concept), np.arange(args.n_test_samples_per_concept)[::args.test_batch_size])
                for chunk_ixs_samples in split_ixs_samples[1:]:
                    eval_in = {'img': imgs[cix][chunk_ixs_samples],
                               'attr': torch.FloatTensor(np.repeat([current_concept['categorical']], len(chunk_ixs_samples), axis=0)).cuda().long()}
                    sample_chunk_out_list.append(ec.test_forward(eval_in))

                # log all results / metrics for the current concept in the respective dicts.
                out_list = [torch.cat([sample_chunk_out_list[chix][aix] for chix in range(len(sample_chunk_out_list))]) for aix in range(len(sample_chunk_out_list[0]))]
                n_unspecified = len([ix for ix in current_concept['categorical'] if ix == -1])
                n_specified = len(args.attribute_dict) - n_unspecified
                softmax_over_cats_dict = {}
                attribute_matches = []
                # The classifier returns a list, where each element (ov) are output logits from an attribute-head. Since we calculate metrics
                # on a concept level, we need to store matches with gt per attribute (attribute_matches).
                # Additionally, mean softmax values over the batch dimension (i.e. n_sampled_images) are stored (softmax_over_cats_dict) to calculate the
                # model distribution in the coverage metric.
                save_consistency_preds = []
                for attr_head_ix, attr_head_logits in enumerate(out_list):

                    softmax_over_cats_dict[list(args.attribute_dict.keys())[attr_head_ix]] = F.softmax(attr_head_logits, 1).mean(0) # mean over sampled images.
                    gt = torch.tensor(current_concept['categorical'][attr_head_ix]).cuda().unsqueeze(0).expand(args.n_test_samples_per_concept)
                    # get the argmax vector over batch elements for the current attribute to check the entire concept for correctness later
                    if current_concept['categorical'][attr_head_ix] != -1:
                        save_consistency_preds.append(attr_head_logits.argmax(1))
                        attribute_matches.append((attr_head_logits.argmax(1) == gt).float().unsqueeze(1))

                concept_consistency_for_all_samples = (torch.cat(attribute_matches, 1).sum(1) == len(attribute_matches)).float()
                concept_consistency = concept_consistency_for_all_samples.mean().item()
                attr_consistency = torch.cat(attribute_matches, 1).mean().item()
                # compute final consistency metric only on fully specified concepts.
                if n_unspecified == 0:
                    consistency_cl[current_concept['type']]['value'].append(concept_consistency)
                    consistency_cl[current_concept['type']]['n_unspecified'].append(n_unspecified)
                    consistency_al[current_concept['type']]['value'].append(attr_consistency)
                    consistency_al[current_concept['type']]['n_unspecified'].append(n_unspecified)

                #comb metric
                #This implementation is made to scale to multiple unspecified dimensions. E.g. if color and shape are both
                # unspecified, the model_dist consists of probabilities like p(red,cube) = p(red) * p(cube). The factors of this
                # product are taken from the stored softmax_over_cats_dict.
                model_distribution = []
                for cp in current_concept['comb_metric_points']:
                    concept_prob = 1
                    for dict in cp:
                        for k, v in dict.items():
                            concept_prob *= softmax_over_cats_dict[k][args.attribute_dict[k].index(v)]
                    model_distribution.append(concept_prob)
                reference_distribution_comb = torch.FloatTensor(current_concept['p_k_comb']).cuda()
                model_distribution_comb = torch.FloatTensor(model_distribution).cuda()
                current_comb_score = 1 - get_js_div(reference_distribution_comb, model_distribution_comb).item()
                comb_metric[current_concept['type']]['value'].append(current_comb_score)
                comb_metric[current_concept['type']]['n_unspecified'].append(n_unspecified)


                #coverage
                # This implementation is made to scale to multiple unspecified dimensions. E.g. if color and shape are both
                # unspecified, the model_dist consists of probabilities like p(red,cube) = p(red) * p(cube). The factors of this
                # product are taken from the stored mean_softmax_arrays.
                if len(current_concept['drop_list']) > 0:
                    model_distribution = []
                    for cp in current_concept['coverage_points']:
                        concept_prob = 1
                        for dict in cp:
                            for k, v in dict.items():
                                concept_prob *= softmax_over_cats_dict[k][args.attribute_dict[k].index(v)]
                        model_distribution.append(concept_prob)

                    reference_distribution = torch.FloatTensor(current_concept['p_k']).cuda()
                    model_distribution = torch.FloatTensor(model_distribution).cuda()
                    current_coverage = 1 - get_js_div(reference_distribution, model_distribution).item()
                    coverage[current_concept['type']]['value'].append(current_coverage)
                    coverage[current_concept['type']]['n_unspecified'].append(n_unspecified)

                    current_tvd = (model_distribution - reference_distribution).abs().max().item()
                    total_variation_distance[current_concept['type']]['value'].append(current_tvd)
                    total_variation_distance[current_concept['type']]['n_unspecified'].append(n_unspecified)

                    # compute frequencies.
                    # if 'unseen' in current_concept['type']:
                    #     for dict in args.unseen_concepts[current_concept['type']]:
                    #             concept_prob = 1
                    #             for k, v in dict.items():
                    #                 concept_prob *= softmax_over_cats_dict[k][args.attribute_dict[k].index(v)]
                    #             frequency[current_concept['type']]['value'].append(concept_prob.item())
                    #             frequency[current_concept['type']]['n_unspecified'].append(n_unspecified)

                    if current_concept['plot_dist_flag']:
                        current_concept['q_k_comb'] = model_distribution_comb.cpu().data.numpy()
                        current_concept['comb_metric'] = current_comb_score
                        current_concept['consistency'] = concept_consistency
                        current_concept['coverage'] = current_coverage

                if n_specified == 1:
                    # add representations to list for factor-vae metric
                    fixed_index = np.argwhere(np.array(current_concept['categorical'])!=-1)[0, 0]
                else:
                    fixed_index = -1

    results_dict = {}
    results_dict['write_out'] = {}

    plot_dict = {}
    plot_dict['confusion_matrix'] = confusion_matrix
    plot_dict['dist_concepts'] = [i for i in test_concepts_list if i['plot_dist_flag'] == True]
    plot_dict['concepts'] = [i for i in test_concepts_list if i['plot_flag'] == True]
    results_dict, plot_dict = append_metric_dict(consistency_cl, 'consistency_cl', results_dict, plot_dict)
    results_dict, plot_dict = append_metric_dict(consistency_al, 'consistency_al', results_dict, plot_dict)
    results_dict, plot_dict = append_metric_dict(coverage, 'coverage', results_dict, plot_dict)
    results_dict, plot_dict = append_metric_dict(frequency, 'frequency', results_dict, plot_dict)
    results_dict, plot_dict = append_metric_dict(comb_metric, 'comb_metric', results_dict, plot_dict)
    results_dict, plot_dict = append_metric_dict(total_variation_distance, 'tvd', results_dict, plot_dict)


    results_dict['plot_dict'] = plot_dict
    print({k: v for k, v in results_dict['write_out'].items() if 'list' not in k})

    return results_dict



