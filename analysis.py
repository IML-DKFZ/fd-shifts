from omegaconf import OmegaConf
import os
import numpy as np
from utils.eval_utils import ConfidEvaluator
import pandas as pd



class Analysis():
    def __init__(self, path_list, query_metrics, analysis_out_dir):

        self.input_list = []
        for _ in path_list:
            method_dict = {
                "cfg": OmegaConf.load(os.path.join(os.path.dirname(path_list[0]), "hydra", "config.yaml")),
                "raw_outputs": np.load(os.path.join(path_list[0], "softmax_labels.npy"))
            }
            self.input_list.append(method_dict)

        self.query_metrics = query_metrics
        self.analysis_out_dir = analysis_out_dir
        self.calibration_bins = 20

    def process_outputs(self):

        for method_dict in self.input_list:

            method_dict["softmax"] = method_dict["raw_outputs"][:, :-1]
            method_dict["labels"] = method_dict["raw_outputs"][:, -1]
            # here is where threshold considerations would come int
            # also with BDL methods here first the merging method needs to be decided.
            method_dict["correct"] = (np.argmax(method_dict["softmax"], axis=1) == method_dict["labels"]) * 1

            confid_types = method_dict["cfg"].model.confidence_measures
            method_dict["confids"] = {}
            if "mcp" in confid_types:
                method_dict["confids"]["mcp"] = np.max(method_dict["softmax"], axis=1)
            if "pe" in confid_types:
                method_dict["confids"]["pe"] = np.sum(method_dict["softmax"] * (- np.log(method_dict["softmax"])), axis=1)


    def compute_all_metrics(self):

        for method_dict in self.input_list:
            metrics_dict = {}
            plots_dict = {}

            if "accuracy" in self.query_metrics:
                metrics_dict["accuracy"] = np.sum(method_dict["correct"]) / method_dict["correct"].size

            for confid_key, confids in method_dict["confids"].items():
                eval = ConfidEvaluator(confids=confids,
                                       correct=method_dict["correct"],
                                       query_metrics=self.query_metrics,
                                       bins=self.calibration_bins)

                metrics_dict.update(eval.get_metrics_per_confid())
                plots_dict["default_plot"] = eval.get_default_plot()

            method_dict["metrics"] = metrics_dict
            method_dict["plots"] = plots_dict

    def create_results_csv(self):

        df = pd.DataFrame()


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
                     "aurcs",
                     "fpr@95tpr",
                     ]

    analysis_out_dir = "mnt/hdd2/checkpoints/analysis/check_analysis"

    if not os.path.exists(analysis_out_dir):
        os.makedirs(analysis_out_dir)


    analysis = Analysis(path_list=path_list,
                        query_metrics=query_metrics,
                        analysis_out_dir=analysis_out_dir
                        )

    analysis.process_outputs()
    analysis.compute_all_metrics()
    analysis.create_results_csv()



if __name__ == '__main__':
   main()