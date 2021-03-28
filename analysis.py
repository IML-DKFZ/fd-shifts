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
                "raw_outputs": np.load(os.path.join(path_list[0], "softmax_labels.npy"))
            }
            self.input_list.append(method_dict)

        self.query_metrics = query_metrics
        self.query_plots = query_plots
        self.analysis_out_dir = analysis_out_dir
        self.calibration_bins = 20

    def process_outputs(self):

        for method_dict in self.input_list:

            method_dict["name"] = method_dict["cfg"].model.method
            method_dict["softmax"] = method_dict["raw_outputs"][:, :-1]
            method_dict["labels"] = method_dict["raw_outputs"][:, -1]
            # here is where threshold considerations would come int
            # also with BDL methods here first the merging method needs to be decided.
            method_dict["correct"] = (np.argmax(method_dict["softmax"], axis=1) == method_dict["labels"]) * 1

            confid_types = method_dict["cfg"].model.confidence_measures
            method_dict["confid_types"] = {}
            if "mcp" in confid_types:
                method_dict["confid_types"]["mcp"] = {}
                method_dict["confid_types"]["mcp"]["confids"] = np.max(method_dict["softmax"], axis=1)
            if "pe" in confid_types:
                method_dict["confid_types"]["pe"] = {}
                method_dict["confid_types"]["pe"]["confids"] = np.sum(method_dict["softmax"] * (- np.log(method_dict["softmax"])), axis=1)


    def compute_all_metrics(self):

        for method_dict in self.input_list:

            accuracy = np.sum(method_dict["correct"]) / method_dict["correct"].size

            for confid_key, confid_dict in method_dict["confid_types"].items():

                method_dict[confid_key] = {}

                if confid_key == "pe":
                    min_confid = np.min(confid_dict["confids"])
                    max_confid = np.max(confid_dict["confids"])
                    confid_dict["confids"] = 1 - ((confid_dict["confids"] - min_confid) / (max_confid - min_confid))

                eval = ConfidEvaluator(confids=confid_dict["confids"],
                                       correct=method_dict["correct"],
                                       query_metrics=self.query_metrics,
                                       bins=self.calibration_bins)

                confid_dict["metrics"] = eval.get_metrics_per_confid()
                confid_dict["plot_stats"] = eval.get_plot_stats_per_confid()
                if "accuracy" in self.query_metrics:
                    confid_dict["metrics"]["accuracy"] = accuracy


    def create_results_csv(self):

        columns = ["name", "confid"] + self.query_metrics
        df = pd.DataFrame(columns=columns)
        for method_dict in self.input_list:
            for confid_key in list(method_dict["confid_types"].keys()):
                submit_list = [method_dict["name"], confid_key]
                submit_list+= [method_dict["confid_types"][confid_key]["metrics"][x] for x in self.query_metrics]
                df.loc[len(df)] = submit_list

        # df = df.round(3)
        df.to_csv(os.path.join(self.analysis_out_dir, "metrics.csv"), float_format='%.3f', decimal='.')
        print("saved csv to ", os.path.join(self.analysis_out_dir, "metrics.csv"))


    def create_master_plot(self):
        plotter = ConfidPlotter(self.input_list, self.query_plots, self.calibration_bins)
        f = plotter.compose_plot()
        f.savefig(os.path.join(self.analysis_out_dir, "master_plot.png"))
        print("saved masterplot to ", os.path.join(self.analysis_out_dir, "master_plot.png"))

def main():

    path_list = [
        "/mnt/hdd2/checkpoints/checks/check_raw_output/version_10"
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

    analysis_out_dir = "/mnt/hdd2/checkpoints/analysis/check_analysis"

    if not os.path.exists(analysis_out_dir):
        os.mkdir(analysis_out_dir)


    analysis = Analysis(path_list=path_list,
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