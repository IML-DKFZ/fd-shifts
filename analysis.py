from omegaconf import OmegaConf
import os
import numpy as np
from src.utils.eval_utils import ConfidEvaluator, ConfidPlotter
import pandas as pd
from copy import deepcopy


class Analysis():
    def __init__(self,
                 path_list,
                 query_performance_metrics,
                 query_confid_metrics,
                 query_plots,
                 analysis_out_dir,):

        self.input_list = []
        self.names_list = []
        for path in path_list:
            method_dict = {
                "cfg": OmegaConf.load(os.path.join(os.path.dirname(path), "hydra", "config.yaml")),
                "raw_outputs": np.load(os.path.join(path, "raw_output.npy")),
                "name": path.split("/")[-2] # last level is version or test dir
            }
            if os.path.isfile(os.path.join(path, "external_confids.npy")):
                method_dict["external_confids"] = np.load(os.path.join(path, "external_confids.npy"))
            self.input_list.append(method_dict)


        self.query_performance_metrics = query_performance_metrics
        self.query_confid_metrics = query_confid_metrics
        self.query_plots = query_plots
        self.analysis_out_dir = analysis_out_dir
        self.calibration_bins = 20
        self.num_classes = self.input_list[0]["cfg"].trainer.num_classes

    def process_outputs(self):

        for method_dict in self.input_list:

            softmax = method_dict["raw_outputs"][:, :-1]
            labels = method_dict["raw_outputs"][:, -1]
            query_confids = method_dict["cfg"].eval.confidence_measures["test"]
            if any("mcd" in cfd for cfd in query_confids):
                mcd_softmax_dist = softmax.reshape(softmax.shape[0], method_dict["cfg"].trainer.num_classes, -1)
                mcd_softmax_mean = np.mean(mcd_softmax_dist, axis=2)
                softmax = mcd_softmax_dist[:,:, 0]
                mcd_correct = (np.argmax(mcd_softmax_mean, axis=1) == labels) * 1
                mcd_performance_metrics = self.compute_performance_metrics(mcd_softmax_mean, labels, mcd_correct)

            # now with the first entry of mcd sampels defined as the det softmax
            correct = (np.argmax(softmax, axis=1) == labels) * 1
            performance_metrics = self.compute_performance_metrics(softmax, labels, correct)
            # here is where threshold considerations would come int
            # also with BDL methods here first the merging method needs to be decided.

            # The case that test measures are not in val is prohibited anyway, because mcd-softmax output needs to fit.
            if "det_mcp" in query_confids:
                method_dict["det_mcp"] = {}
                method_dict["det_mcp"]["confids"] = np.max(softmax, axis=1)
                method_dict["det_mcp"]["correct"] = correct
                method_dict["det_mcp"]["metrics"] = deepcopy(performance_metrics)

            if "det_pe" in query_confids:
                method_dict["det_pe"] = {}
                method_dict["det_pe"]["confids"] = np.sum(softmax * (- np.log(softmax)), axis=1)
                method_dict["det_pe"]["correct"] = correct
                method_dict["det_pe"]["metrics"] = deepcopy(performance_metrics)

            if "mcd_mcp" in query_confids:
                method_dict["mcd_mcp"] = {}
                tmp_confids = np.max(mcd_softmax_mean, axis=1)
                method_dict["mcd_mcp"]["confids"] = tmp_confids
                method_dict["mcd_mcp"]["correct"] = mcd_correct
                method_dict["mcd_mcp"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_pe" in query_confids:
                method_dict["mcd_pe"] = {}
                tmp_confids = np.sum(mcd_softmax_mean *
                                     (- np.log(mcd_softmax_mean)), axis=1)
                method_dict["mcd_pe"]["confids"] = tmp_confids
                method_dict["mcd_pe"]["correct"] = mcd_correct
                method_dict["mcd_pe"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_ee" in query_confids:
                method_dict["mcd_ee"] = {}
                tmp_confids = np.mean(np.sum(mcd_softmax_dist *
                                             (- np.log(mcd_softmax_dist)), axis=1), axis=1)
                method_dict["mcd_ee"]["confids"] = tmp_confids
                method_dict["mcd_ee"]["correct"] = mcd_correct
                method_dict["mcd_ee"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_mi" in query_confids:
                method_dict["mcd_mi"] = {}
                tmp_confids = method_dict["mcd_pe"]["confids"]-method_dict["mcd_ee"]["confids"]
                method_dict["mcd_mi"]["confids"] = tmp_confids
                method_dict["mcd_mi"]["correct"] = mcd_correct
                method_dict["mcd_mi"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_sv" in query_confids:
                method_dict["mcd_sv"] = {}
                # [b, cl, mcd] - [b, cl]
                tmp_confids = np.mean((mcd_softmax_dist - np.expand_dims(mcd_softmax_mean, axis=2))**2, axis=(1,2))
                method_dict["mcd_sv"]["confids"] = tmp_confids
                method_dict["mcd_sv"]["correct"] = mcd_correct
                method_dict["mcd_sv"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "tcp" in query_confids:
                method_dict["tcp"] = {}
                # [b, cl, mcd] - [b, cl]
                method_dict["tcp"]["confids"] = method_dict["external_confids"]
                method_dict["tcp"]["correct"] = correct
                method_dict["tcp"]["metrics"] = deepcopy(performance_metrics)

            method_dict["query_confids"] = query_confids

    def compute_performance_metrics(self, softmax, labels, correct):

        performance_metrics = {}
        y_one_hot = np.eye(self.num_classes)[labels.astype("int")]
        if "nll" in self.query_performance_metrics:
            performance_metrics["nll"] = np.mean(- np.log(softmax) * y_one_hot)
        if "accuracy" in self.query_performance_metrics:
            performance_metrics["accuracy"] = np.sum(correct) / correct.size
        if "brier_score" in self.query_performance_metrics:
            # [b, classes]
            mse = (softmax - y_one_hot) ** 2
            performance_metrics["brier_score"] = np.mean(np.sum(mse, axis=1))

        return performance_metrics

    def compute_confid_metrics(self):

        for ix, method_dict in enumerate(self.input_list):

            for confid_key in method_dict["query_confids"]:
                print(confid_key)
                confid_dict = method_dict[confid_key]
                if any(cfd in confid_key for cfd  in ["_pe", "_ee", "_mi", "_sv"]):
                    min_confid = np.min(confid_dict["confids"])
                    max_confid = np.max(confid_dict["confids"])
                    confid_dict["confids"] = 1 - ((confid_dict["confids"] - min_confid) / (max_confid - min_confid))

                eval = ConfidEvaluator(confids=confid_dict["confids"],
                                       correct=confid_dict["correct"],
                                       query_metrics=self.query_confid_metrics,
                                       query_plots=self.query_plots,
                                       bins=self.calibration_bins)

                confid_dict["metrics"].update(eval.get_metrics_per_confid())
                confid_dict["plot_stats"] = eval.get_plot_stats_per_confid()


    def create_results_csv(self):

        all_metrics = self.query_performance_metrics + self.query_confid_metrics
        columns = ["name", "model", "confid"] + all_metrics
        df = pd.DataFrame(columns=columns)
        for method_dict in self.input_list:
            for confid_key in method_dict["query_confids"]:
                submit_list = [method_dict["name"], method_dict["cfg"].model.name, confid_key]
                submit_list+= [method_dict[confid_key]["metrics"][x] for x in all_metrics]
                df.loc[len(df)] = submit_list

        df.to_csv(os.path.join(self.analysis_out_dir, "analysis_metrics.csv"), float_format='%.3f', decimal='.')
        print("saved csv to ", os.path.join(self.analysis_out_dir, "analysis_metrics.csv"))


    def create_master_plot(self):
        # get overall with one dict per compared_method (i.e confid)
        input_dict = {"{}_{}".format(method_dict["name"], k):method_dict[k] for method_dict in self.input_list for k in method_dict["query_confids"] }
        plotter = ConfidPlotter(input_dict, self.query_plots, self.calibration_bins)
        f = plotter.compose_plot()
        f.savefig(os.path.join(self.analysis_out_dir, "master_plot.png"))
        print("saved masterplot to ", os.path.join(self.analysis_out_dir, "master_plot.png"))




def main(in_path=None, out_path=None):

    # path to the dir where the raw otuputs lie. NO SLASH AT THE END!
    if in_path is None:
        path_to_test_dir_list = [
            "/mnt/hdd2/checkpoints/checks/repro_mcd_mcp_3/test_results",
        ]
        # path_to_test_dir_list = [
        #     "/gpu/checkpoints/OE0612/jaegerp/checks/check_mcd/fold_0/version_0",
        #     "/gpu/checkpoints/OE0612/jaegerp/checks/check_mcd/fold_1/version_0",
        #     "/gpu/checkpoints/OE0612/jaegerp/checks/check_mcd/fold_2/version_0",
        #     "/gpu/checkpoints/OE0612/jaegerp/checks/check_mcd/fold_3/version_0",
        #     "/gpu/checkpoints/OE0612/jaegerp/checks/check_mcd/fold_4/version_0",
        # ]
    else:
        path_to_test_dir_list = [in_path]

    if out_path is None:
        # analysis_out_dir = "/mnt/hdd2/checkpoints/analysis/check_analysis_final"
        analysis_out_dir = "/mnt/hdd2/checkpoints/checks/repro_mcd_mcp_3/test_results"
    else:
        analysis_out_dir = out_path


    query_performance_metrics = ['accuracy', 'nll', 'brier_score']
    query_confid_metrics = ['failauc',
                            'failap_suc',
                            'failap_err',
                            "mce",
                            "ece",
                            "e-aurc",
                            "aurc",
                            "fpr@95tpr",
                            ]

    query_plots = ["calibration",
                   "overconfidence",
                   "roc_curve",
                   "prc_curve",
                   "rc_curve",
                   "hist_per_confid"]


    if not os.path.exists(analysis_out_dir):
        os.mkdir(analysis_out_dir)

    print("starting analysis with in_path(s) {} and out_path {}".format(path_to_test_dir_list, analysis_out_dir))

    analysis = Analysis(path_list=path_to_test_dir_list,
                        query_performance_metrics=query_performance_metrics,
                        query_confid_metrics=query_confid_metrics,
                        query_plots=query_plots,
                        analysis_out_dir=analysis_out_dir
                        )

    analysis.process_outputs()
    analysis.compute_confid_metrics()
    analysis.create_results_csv()
    analysis.create_master_plot()



if __name__ == '__main__':
   main()