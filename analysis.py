from omegaconf import OmegaConf
import os
import numpy as np
from src.utils.eval_utils import ConfidEvaluator, ConfidPlotter
import pandas as pd



class Analysis():
    def __init__(self,
                 path_list,
                 query_metrics,
                 query_plots,
                 analysis_out_dir,):

        self.input_list = []
        for _ in path_list:
            method_dict = {
                "cfg": OmegaConf.load(os.path.join(os.path.dirname(path_list[0]), "hydra", "config.yaml")),
                "raw_outputs": np.load(os.path.join(path_list[0], "raw_output.npy"))
            }
            self.input_list.append(method_dict)

        self.query_metrics = query_metrics
        self.query_plots = query_plots
        self.analysis_out_dir = analysis_out_dir
        self.calibration_bins = 20

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

            correct = (np.argmax(softmax, axis=1) == labels) * 1
            # here is where threshold considerations would come int
            # also with BDL methods here first the merging method needs to be decided.

            # The case that test measures are not in val is prohibited anyway, because mcd-softmax output needs to fit.
            if "det_mcp" in query_confids:
                method_dict["det_mcp"] = {}
                method_dict["det_mcp"]["confids"] = np.max(softmax, axis=1)
                method_dict["det_mcp"]["correct"] = correct

            if "det_pe" in query_confids:
                method_dict["det_pe"] = {}
                method_dict["det_pe"]["confids"] = np.sum(softmax * (- np.log(softmax)), axis=1)
                method_dict["det_pe"]["correct"] = correct

            if "mcd_mcp" in query_confids:
                method_dict["mcd_mcp"] = {}
                tmp_confids = np.max(mcd_softmax_mean, axis=1)
                method_dict["mcd_mcp"]["confids"] = tmp_confids
                method_dict["mcd_mcp"]["correct"] = mcd_correct

            if "mcd_pe" in query_confids:
                method_dict["mcd_pe"] = {}
                tmp_confids = np.sum(mcd_softmax_mean *
                                     (- np.log(mcd_softmax_mean)), axis=1)
                method_dict["mcd_pe"]["confids"] = tmp_confids
                method_dict["mcd_pe"]["correct"] = mcd_correct

            if "mcd_ee" in query_confids:
                method_dict["mcd_ee"] = {}
                tmp_confids = np.mean(np.sum(mcd_softmax_dist *
                                             (- np.log(mcd_softmax_dist)), axis=1), axis=1)
                method_dict["mcd_ee"]["confids"] = tmp_confids
                method_dict["mcd_ee"]["correct"] = mcd_correct


            # todo REMOVE
            query_confids += ["mcd_mi", "mcd_sv"]

            if "mcd_mi" in query_confids:
                method_dict["mcd_mi"] = {}
                tmp_confids = method_dict["mcd_pe"]["confids"]-method_dict["mcd_ee"]["confids"]
                method_dict["mcd_mi"]["confids"] = tmp_confids
                method_dict["mcd_mi"]["correct"] = mcd_correct

            if "mcd_sv" in query_confids:
                method_dict["mcd_sv"] = {}
                # [b, cl, mcd] - [b, cl]
                tmp_confids = np.mean((mcd_softmax_dist - np.expand_dims(mcd_softmax_mean, axis=2))**2, axis=(1,2))
                method_dict["mcd_sv"]["confids"] = tmp_confids
                method_dict["mcd_sv"]["correct"] = mcd_correct

            method_dict["query_confids"] = query_confids


    def compute_all_metrics(self):

        for method_dict in self.input_list:

            for confid_key in method_dict["query_confids"]:

                confid_dict = method_dict[confid_key]

                if "_pe" in confid_key or "_ee" in confid_key:
                    min_confid = np.min(confid_dict["confids"])
                    max_confid = np.max(confid_dict["confids"])
                    confid_dict["confids"] = 1 - ((confid_dict["confids"] - min_confid) / (max_confid - min_confid))

                eval = ConfidEvaluator(confids=confid_dict["confids"],
                                       correct=confid_dict["correct"],
                                       query_metrics=self.query_metrics,
                                       bins=self.calibration_bins)

                confid_dict["metrics"] = eval.get_metrics_per_confid()
                confid_dict["plot_stats"] = eval.get_plot_stats_per_confid()


    def create_results_csv(self):

        columns = ["name", "confid"] + self.query_metrics
        df = pd.DataFrame(columns=columns)
        for method_dict in self.input_list:
            for confid_key in method_dict["query_confids"]:
                submit_list = [method_dict["cfg"].model.name, confid_key]
                submit_list+= [method_dict[confid_key]["metrics"][x] for x in self.query_metrics]
                df.loc[len(df)] = submit_list

        df.to_csv(os.path.join(self.analysis_out_dir, "metrics.csv"), float_format='%.3f', decimal='.')
        print("saved csv to ", os.path.join(self.analysis_out_dir, "metrics.csv"))


    def create_master_plot(self):
        # get overall with one dict per compared_method (i.e confid)
        input_dict = {k:method_dict[k] for method_dict in self.input_list for k in method_dict["query_confids"] }
        plotter = ConfidPlotter(input_dict, self.query_plots, self.calibration_bins)
        f = plotter.compose_plot()
        f.savefig(os.path.join(self.analysis_out_dir, "master_plot.png"))
        print("saved masterplot to ", os.path.join(self.analysis_out_dir, "master_plot.png"))

def main():

    path_to_test_dir_list = [
        "/mnt/hdd2/checkpoints/checks/check_mcs/test_results",
    ]

    query_metrics = ['accuracy',
                     'failauc',
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

    analysis_out_dir = "/mnt/hdd2/checkpoints/analysis/check_analysis_dropout"

    if not os.path.exists(analysis_out_dir):
        os.mkdir(analysis_out_dir)


    analysis = Analysis(path_list=path_to_test_dir_list,
                        query_metrics=query_metrics,
                        query_plots=query_plots,
                        analysis_out_dir=analysis_out_dir
                        )

    analysis.process_outputs()
    analysis.compute_all_metrics()
    analysis.create_results_csv()
    analysis.create_master_plot()



if __name__ == '__main__':
   main()