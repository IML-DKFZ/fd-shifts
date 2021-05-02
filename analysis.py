from omegaconf import OmegaConf
import os
import numpy as np
from src.utils.eval_utils import ConfidEvaluator, ConfidPlotter
import pandas as pd
from copy import deepcopy
import omegaconf


class Analysis():
    def __init__(self,
                 path_list,
                 query_performance_metrics,
                 query_confid_metrics,
                 query_plots,
                 query_studies,
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
                method_dict["raw_external_confids"] = np.load(os.path.join(path, "external_confids.npy"))
            if method_dict["cfg"].data.num_classes is None:
                method_dict["cfg"].data.num_classes = method_dict["cfg"].trainer.num_classes
            method_dict["query_confids"] = method_dict["cfg"].eval.confidence_measures["test"]
            self.input_list.append(method_dict)


        self.query_performance_metrics = query_performance_metrics
        self.query_confid_metrics = query_confid_metrics
        self.query_plots = query_plots
        self.query_studies = query_studies
        self.analysis_out_dir = analysis_out_dir
        self.calibration_bins = 20
        self.num_classes = self.input_list[0]["cfg"].data.num_classes



    def process_outputs(self):

        for method_dict in self.input_list:

            raw_outputs = method_dict["raw_outputs"]
            dataset_ix = raw_outputs[:, -1]
            softmax = raw_outputs[:, :-2]
            print("analysis softmax in shape:", softmax.shape)
            labels = raw_outputs[:, -2]

            if any("mcd" in cfd for cfd in method_dict["query_confids"]):
                mcd_softmax_dist = softmax.reshape(softmax.shape[0], method_dict["cfg"].data.num_classes, -1)
                mcd_softmax_mean = np.mean(mcd_softmax_dist, axis=2)
                softmax = mcd_softmax_dist[:,:, 0]
                mcd_correct = (np.argmax(mcd_softmax_mean, axis=1) == labels) * 1
                method_dict["raw_mcd_correct"] = mcd_correct
                method_dict["raw_mcd_softmax_mean"] = mcd_softmax_mean
                method_dict["raw_mcd_softmax_dist"] = mcd_softmax_dist

            # now with the first entry of mcd sampels defined as the det softmax
            correct = (np.argmax(softmax, axis=1) == labels) * 1

            method_dict["raw_softmax"] = softmax
            method_dict["raw_labels"] = labels
            method_dict["raw_correct"] = correct
            method_dict["raw_dataset_ix"] = dataset_ix
            method_dict["raw_external_confids"] = method_dict.get("raw_external_confids")


    def register_and_perform_studies(self):


        self.process_outputs()
        flat_test_set_list = []
        for k, v in self.query_studies.items():
            print("QUERY STUDIES", self.query_studies)
            print(k, v, isinstance(v, list), type(v))
            if isinstance(v, list) or isinstance(v, omegaconf.listconfig.ListConfig):
                flat_test_set_list.extend([dataset for dataset in v])
            else:
                flat_test_set_list.append(v)
        print("CHECK flat list of all test datasets", flat_test_set_list)

        for study_name in self.query_studies.keys():

            if study_name == "val_tuning":
                pass

            # todo val tuning threhold study here.
            # get code from devries. outputs an optimal threshold which will later be used to compute extra metrics in all other studies.

            if study_name == "iid_study":

                iid_set_ix = flat_test_set_list.index(self.query_studies[study_name])
                for method_dict in self.input_list:
                    val_ix_shift = 1 if method_dict["cfg"].eval.val_tuning else 0

                    select_ix = np.argwhere(method_dict["raw_dataset_ix"] == iid_set_ix + val_ix_shift)[:, 0]

                    method_dict["study_softmax"] = deepcopy(method_dict["raw_softmax"][select_ix])
                    method_dict["study_labels"] = deepcopy(method_dict["raw_labels"][select_ix])
                    method_dict["study_correct"] = deepcopy(method_dict["raw_correct"][select_ix])
                    if method_dict["raw_external_confids"] is not None:
                        method_dict["study_external_confids"] = deepcopy(method_dict["raw_external_confids"][select_ix])
                    else:
                        method_dict["study_external_confids"] = None

                    if any("mcd" in cfd for cfd in method_dict["query_confids"]):
                        method_dict["study_mcd_softmax_mean"] = deepcopy(method_dict["raw_mcd_softmax_mean"][select_ix])
                        method_dict["study_mcd_softmax_dist"] = deepcopy(method_dict["raw_mcd_softmax_dist"][select_ix])
                        method_dict["study_mcd_correct"] = deepcopy(method_dict["raw_mcd_correct"][select_ix])

                self.perform_study(study_name)


            if study_name == "new_class_study":


                for method_dict in self.input_list:
                    val_ix_shift = 1 if method_dict["cfg"].eval.val_tuning else 0

                    for new_class_set in self.query_studies[study_name]:
                        for mode in ["original_mode", "proposed_mode"]:

                            iid_set_ix = flat_test_set_list.index(self.query_studies["iid_study"]) + val_ix_shift
                            new_class_set_ix = flat_test_set_list.index(new_class_set) + val_ix_shift
                            select_ix_in = np.argwhere(method_dict["raw_dataset_ix"] == iid_set_ix)[:, 0]
                            select_ix_out = np.argwhere(method_dict["raw_dataset_ix"] == new_class_set_ix)[:, 0]

                            correct = deepcopy(method_dict["raw_correct"])
                            correct[select_ix_out] = 0
                            if mode == "original_mode":
                                correct[select_ix_in] = 1     # nice to see so visual how little practical sense the current protocol makes!
                            labels = deepcopy(method_dict["raw_labels"])
                            labels[select_ix_out] = - 99

                            select_ix_all = np.argwhere((method_dict["raw_dataset_ix"] == new_class_set_ix) | ((method_dict["raw_dataset_ix"] == iid_set_ix) & (correct == 1)))[:, 0] # de-select incorrect inlier predictions.
                            method_dict["study_softmax"] = deepcopy(method_dict["raw_softmax"])[select_ix_all]
                            method_dict["study_labels"] = labels[select_ix_all]
                            method_dict["study_correct"] = correct[select_ix_all]
                            if method_dict["raw_external_confids"] is not None:
                                method_dict["study_external_confids"] = method_dict["raw_external_confids"][select_ix_all]
                            if any("mcd" in cfd for cfd in method_dict["query_confids"]):
                                correct = deepcopy(method_dict["raw_mcd_correct"])
                                correct[select_ix_out] = 0
                                if mode == "original_mode":
                                    correct[select_ix_in] = 1

                                select_ix_all = np.argwhere((method_dict["raw_dataset_ix"] == new_class_set_ix) | ((method_dict["raw_dataset_ix"] == iid_set_ix) & (correct == 1)))[:, 0]
                                method_dict["study_mcd_softmax_mean"] = deepcopy(method_dict["raw_mcd_softmax_mean"][select_ix_all])
                                method_dict["study_mcd_softmax_dist"] = deepcopy(method_dict["raw_mcd_softmax_dist"][select_ix_all])
                                method_dict["study_mcd_correct"] = correct[select_ix_all]

                            self.perform_study(study_name="{}_{}_{}".format(study_name, new_class_set, mode))


            if study_name == "noise_study":

                for noise_set in self.query_studies[study_name]:


                    # here could be a loop over explicit corruptions.
                    for intensity_level in range(5):

                        for method_dict in self.input_list:
                            val_ix_shift = 1 if method_dict["cfg"].eval.val_tuning else 0
                            noise_set_ix = flat_test_set_list.index(noise_set) + val_ix_shift

                            select_ix = np.argwhere(method_dict["raw_dataset_ix"] == noise_set_ix)[:, 0]

                            # new shape: n_corruptions, n_intensity_levels, n_test_cases, n_classes
                            method_dict["study_softmax"] = deepcopy(method_dict["raw_softmax"][select_ix]).reshape(
                                15, 5, -1, method_dict["raw_softmax"].shape[-1])[:, intensity_level].reshape(
                                -1, method_dict["raw_softmax"].shape[-1])
                            method_dict["study_labels"] = deepcopy(method_dict["raw_labels"][select_ix]).reshape(
                                15, 5, -1)[:, intensity_level].reshape(-1)
                            method_dict["study_correct"] = deepcopy(method_dict["raw_correct"][select_ix]).reshape(
                                15, 5, -1)[:, intensity_level].reshape(-1)
                            if method_dict["raw_external_confids"] is not None:
                                method_dict["study_external_confids"] = deepcopy(
                                    method_dict["raw_external_confids"][select_ix]).reshape(15, 5, -1)[:, intensity_level].reshape(-1)
                            else:
                                method_dict["study_external_confids"] = None

                            if any("mcd" in cfd for cfd in method_dict["query_confids"]):
                                method_dict["study_mcd_softmax_mean"] = deepcopy(
                                    method_dict["raw_mcd_softmax_mean"][select_ix]).reshape(
                                    15, 5, -1, method_dict["raw_mcd_softmax_mean"].shape[-1])[:, intensity_level].reshape(
                                    -1, method_dict["raw_mcd_softmax_mean"].shape[-1])
                                method_dict["study_mcd_softmax_dist"] = deepcopy(
                                    method_dict["raw_mcd_softmax_dist"][select_ix]).reshape(
                                    15, 5, -1, method_dict["raw_mcd_softmax_dist"].shape[-2], method_dict["raw_mcd_softmax_dist"].shape[-1])[:, intensity_level].reshape(
                                    -1, method_dict["raw_mcd_softmax_dist"].shape[-2], method_dict["raw_mcd_softmax_dist"].shape[-1])
                                method_dict["study_mcd_correct"] = deepcopy(method_dict["raw_mcd_correct"][select_ix]).reshape(
                                    15, 5, -1)[:, intensity_level].reshape(-1)

                        print("starting noise study with intensitiy level ", intensity_level + 1)
                        self.perform_study(study_name="{}_{}".format(study_name, intensity_level + 1))





    def perform_study(self, study_name):

        self.study_name = study_name
        self.get_confidence_scores()
        self.compute_confid_metrics()
        self.create_results_csv()
        self.create_master_plot()

    # required input to analysis: softmax, labels, correct, mcd_correct, mcd_softmax_mean, mcd_softmax_dist
    # query_confids per method!Method should have a different method now with

    def get_confidence_scores(self):

        for method_dict in self.input_list:

            softmax = method_dict["study_softmax"]
            labels = method_dict["study_labels"]
            correct = method_dict["study_correct"]
            external_confids = method_dict["study_external_confids"]

            if any("mcd" in cfd for cfd in method_dict["query_confids"]):
                mcd_softmax_mean = method_dict["study_mcd_softmax_mean"]
                mcd_softmax_dist = method_dict["study_mcd_softmax_dist"]
                mcd_correct = method_dict["study_mcd_correct"]
                mcd_performance_metrics = self.compute_performance_metrics(mcd_softmax_mean, labels, mcd_correct)

            performance_metrics = self.compute_performance_metrics(softmax, labels, correct)
            # here is where threshold considerations would come int
            # also with BDL methods here first the merging method needs to be decided.

            # The case that test measures are not in val is prohibited anyway, because mcd-softmax output needs to fit.
            if "det_mcp" in method_dict["query_confids"]:
                method_dict["det_mcp"] = {}
                method_dict["det_mcp"]["confids"] = np.max(softmax, axis=1)
                method_dict["det_mcp"]["correct"] = deepcopy(correct)
                method_dict["det_mcp"]["metrics"] = deepcopy(performance_metrics)

            if "det_pe" in method_dict["query_confids"]:
                method_dict["det_pe"] = {}
                method_dict["det_pe"]["confids"] = np.sum(softmax * (- np.log(softmax + 1e-7)), axis=1)
                method_dict["det_pe"]["correct"] = deepcopy(correct)
                method_dict["det_pe"]["metrics"] = deepcopy(performance_metrics)

            if "mcd_mcp" in method_dict["query_confids"]:
                method_dict["mcd_mcp"] = {}
                tmp_confids = np.max(mcd_softmax_mean, axis=1)
                method_dict["mcd_mcp"]["confids"] = tmp_confids
                method_dict["mcd_mcp"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_mcp"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_pe" in method_dict["query_confids"]:
                method_dict["mcd_pe"] = {}
                tmp_confids = np.sum(mcd_softmax_mean *
                                     (- np.log(mcd_softmax_mean + 1e-7)), axis=1)
                method_dict["mcd_pe"]["confids"] = tmp_confids
                method_dict["mcd_pe"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_pe"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_ee" in method_dict["query_confids"]:
                method_dict["mcd_ee"] = {}
                tmp_confids = np.mean(np.sum(mcd_softmax_dist *
                                             (- np.log(mcd_softmax_dist + 1e-7)), axis=1), axis=1)
                method_dict["mcd_ee"]["confids"] = tmp_confids
                method_dict["mcd_ee"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_ee"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_mi" in method_dict["query_confids"]:
                method_dict["mcd_mi"] = {}
                tmp_confids = method_dict["mcd_pe"]["confids"]-method_dict["mcd_ee"]["confids"]
                method_dict["mcd_mi"]["confids"] = tmp_confids
                method_dict["mcd_mi"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_mi"]["metrics"] = deepcopy(mcd_performance_metrics)

            if "mcd_sv" in method_dict["query_confids"]:
                method_dict["mcd_sv"] = {}
                # [b, cl, mcd] - [b, cl]
                tmp_confids = np.mean((mcd_softmax_dist - np.expand_dims(mcd_softmax_mean, axis=2))**2, axis=(1,2))
                method_dict["mcd_sv"]["confids"] = tmp_confids
                method_dict["mcd_sv"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_sv"]["metrics"] = deepcopy(mcd_performance_metrics)

            if any(cfd in method_dict["query_confids"] for cfd  in ["ext", "bpd", "tcp", "devries"]):
                ext_confid_name = method_dict["cfg"].eval.ext_confid_name
                method_dict[ext_confid_name] = {}
                # [b, cl, mcd] - [b, cl]
                method_dict[ext_confid_name]["confids"] = external_confids
                method_dict[ext_confid_name]["correct"] = deepcopy(correct)
                method_dict[ext_confid_name]["metrics"] = deepcopy(performance_metrics)
                method_dict["query_confids"] = [ext_confid_name  if v=="ext" else v for v in method_dict["query_confids"]]


    def compute_performance_metrics(self, softmax, labels, correct):
        performance_metrics = {}
        if "nll" in self.query_performance_metrics:
            if "new_class" in self.study_name:
                performance_metrics["nll"] = None
            else:
                y_one_hot = np.eye(self.num_classes)[labels.astype("int")]
                performance_metrics["nll"] = np.mean(- np.log(softmax + 1e-7) * y_one_hot)
        if "accuracy" in self.query_performance_metrics:
            performance_metrics["accuracy"] = np.sum(correct) / correct.size
        if "brier_score" in self.query_performance_metrics:
            if "new_class" in self.study_name:
                performance_metrics["brier_score"] = None
            else:
                y_one_hot = np.eye(self.num_classes)[labels.astype("int")] # [b, classes]
                mse = (softmax - y_one_hot) ** 2
                performance_metrics["brier_score"] = np.mean(np.sum(mse, axis=1))

        return performance_metrics

    def compute_confid_metrics(self):

        for ix, method_dict in enumerate(self.input_list):

            for confid_key in method_dict["query_confids"]:
                print(method_dict.keys())
                confid_dict = method_dict[confid_key]
                if confid_key == "bpd":
                    print("CHECK BEFORE NORM VALUES CORRECT", np.median(confid_dict["confids"][confid_dict["correct"] == 1]))
                    print("CHECK BEFORE NORM VALUES INCORRECT", np.median(confid_dict["confids"][confid_dict["correct"] == 0]))
                if any(cfd in confid_key for cfd  in ["_pe", "_ee", "_mi", "_sv", "bpd"]):
                    confids = confid_dict["confids"].astype(np.float64)
                    min_confid = np.min(confids)
                    max_confid = np.max(confids)
                    confid_dict["confids"] = 1 - ((confids - min_confid) / (max_confid - min_confid))

                if confid_key == "bpd":
                    print("CHECK AFTER NORM VALUES CORRECT", np.median(confid_dict["confids"][confid_dict["correct"] == 1]))
                    print("CHECK AFTER NORM VALUES INCORRECT", np.median(confid_dict["confids"][confid_dict["correct"] == 0]))

                eval = ConfidEvaluator(confids=confid_dict["confids"],
                                       correct=confid_dict["correct"],
                                       query_metrics=self.query_confid_metrics,
                                       query_plots=self.query_plots,
                                       bins=self.calibration_bins)

                confid_dict["metrics"].update(eval.get_metrics_per_confid())
                confid_dict["plot_stats"] = eval.get_plot_stats_per_confid()


    def create_results_csv(self):

        all_metrics = self.query_performance_metrics + self.query_confid_metrics
        columns = ["name", "study", "model", "network", "fold", "confid", "n_test"] + all_metrics
        df = pd.DataFrame(columns=columns)
        for method_dict in self.input_list:
            for confid_key in method_dict["query_confids"]:
                submit_list = [method_dict["name"],
                               self.study_name,
                               method_dict["cfg"].model.name,
                               method_dict["cfg"].model.network.backbone,
                               method_dict["cfg"].exp.fold,
                               confid_key,
                               method_dict["study_mcd_softmax_mean"].shape[0] if "mcd" in confid_key else method_dict["study_softmax"].shape[0]]
                submit_list+= [method_dict[confid_key]["metrics"][x] for x in all_metrics]
                df.loc[len(df)] = submit_list
        print("CHECK SHIFT", self.study_name, all_metrics, self.input_list[0]["det_mcp"].keys())
        df.to_csv(os.path.join(self.analysis_out_dir, "analysis_metrics_{}.csv").format(self.study_name), float_format='%.5f', decimal='.')
        print("saved csv to ", os.path.join(self.analysis_out_dir, "analysis_metrics_{}.csv".format(self.study_name)))

        group_file_path = os.path.join(self.input_list[0]["cfg"].exp.group_dir, "group_analysis_metrics.csv")
        if os.path.exists(group_file_path):
            with open(group_file_path, 'a') as f:
                df.to_csv(f, float_format='%.5f', decimal='.', header=False)
        else:
            with open(group_file_path, 'w') as f:
                df.to_csv(f, float_format='%.5f', decimal='.')



    def create_master_plot(self):
        # get overall with one dict per compared_method (i.e confid)
        input_dict = {"{}_{}".format(method_dict["name"], k):method_dict[k] for method_dict in self.input_list for k in method_dict["query_confids"] }
        plotter = ConfidPlotter(input_dict, self.query_plots, self.calibration_bins, fig_scale=1) # fig_scale big: 5
        f = plotter.compose_plot()
        f.savefig(os.path.join(self.analysis_out_dir, "master_plot_{}.png".format(self.study_name)))
        print("saved masterplot to ", os.path.join(self.analysis_out_dir, "master_plot_{}.png".format(self.study_name)))




def main(in_path=None, out_path=None, query_studies=None):

    # path to the dir where the raw otuputs lie. NO SLASH AT THE END!
    if in_path is None: # NO SLASH AT THE END OF PATH !
        path_to_test_dir_list = [
            # "/mnt/hdd2/checkpoints/checks/check_mnist/test_results",
            "/mnt/hdd2/checkpoints/checks/check_devries_multilr/test_results",
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
        # analysis_out_dir = "/mnt/hdd2/checkpoints/checks/check_confidnet_their_backbone/test_results"
        analysis_out_dir = path_to_test_dir_list[0]
    else:
        analysis_out_dir = out_path


    if query_studies is None:
        print("Analysis input query studies was None, setting to hardcoded studies.")
        query_studies = {"iid_study": "svhn", "new_class_study": ["cifar10"]}

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



    print("starting analysis with in_path(s) {}, out_path {}, and query studies {}".format(path_to_test_dir_list, analysis_out_dir, query_studies))


    analysis = Analysis(path_list=path_to_test_dir_list,
                        query_performance_metrics=query_performance_metrics,
                        query_confid_metrics=query_confid_metrics,
                        query_plots=query_plots,
                        query_studies=query_studies,
                        analysis_out_dir=analysis_out_dir
                        )

    analysis.register_and_perform_studies()



if __name__ == '__main__':
   main()