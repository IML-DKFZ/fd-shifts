from omegaconf import OmegaConf
import os
import numpy as np
from src.utils.eval_utils import ConfidEvaluator, ConfidPlotter, ThresholdPlot, qual_plot, cifar100_classes
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
                 analysis_out_dir,
                 add_val_tuning,
                 threshold_plot_confid,
                 qual_plot_confid,
                 cf):

        self.input_list = []
        self.names_list = []
        for path in path_list:
            method_dict = {
                "cfg": OmegaConf.load(os.path.join(os.path.dirname(path), "hydra", "config.yaml")) if cf is None else cf,
                "name": path.split("/")[-2] # last level is version or test dir
            }
            for np_postfix in ["npy", "npz", "npy.npz"]:
                if os.path.isfile(os.path.join(path, "raw_output.{}".format(np_postfix))):
                    method_dict["raw_output"] = np.load(os.path.join(path, "raw_output.{}".format(np_postfix)))
                if os.path.isfile(os.path.join(path, "raw_output_dist.{}".format(np_postfix))):
                    method_dict["raw_output_dist"] = np.load(os.path.join(path, "raw_output_dist.{}".format(np_postfix)))
                if os.path.isfile(os.path.join(path, "external_confids.{}".format(np_postfix))):
                    method_dict["raw_external_confids"] = np.load(os.path.join(path, "external_confids.{}".format(np_postfix)))
                if os.path.isfile(os.path.join(path, "external_confids_dist.{}".format(np_postfix))):
                    method_dict["raw_external_confids_dist"] = np.load(os.path.join(path, "external_confids_dist.{}".format(np_postfix)))

            if method_dict["cfg"].data.num_classes is None:
                method_dict["cfg"].data.num_classes = method_dict["cfg"].trainer.num_classes
            method_dict["query_confids"] = method_dict["cfg"].eval.confidence_measures["test"]
            print("CHECK QUERY CONFIDS", method_dict["query_confids"])
            self.input_list.append(method_dict)


        self.query_performance_metrics = query_performance_metrics
        self.query_confid_metrics = query_confid_metrics
        self.query_plots = query_plots
        self.query_studies = self.input_list[0]["cfg"].eval.query_studies if query_studies is None else query_studies
        self.analysis_out_dir = analysis_out_dir
        self.calibration_bins = 20
        self.val_risk_scores = {}
        self.num_classes = self.input_list[0]["cfg"].data.num_classes
        self.add_val_tuning = add_val_tuning
        self.threshold_plot_confid = threshold_plot_confid
        self.qual_plot_confid = qual_plot_confid



    def process_outputs(self):

        for method_dict in self.input_list:

            raw_outputs = method_dict["raw_output"]
            try:
                raw_outputs = raw_outputs.f.arr_0
            except:
                pass
            method_dict["raw_dataset_ix"] = raw_outputs[:, -1]
            print("CHECK IN DATASETS", np.unique(method_dict["raw_dataset_ix"], return_counts=True))
            method_dict["raw_labels"] = raw_outputs[:, -2]
            method_dict["raw_softmax"] = raw_outputs[:, :-2]
            method_dict["raw_correct"] = (np.argmax(method_dict["raw_softmax"], axis=1) == method_dict["raw_labels"]) * 1
            print("analysis softmax in shape:", method_dict["raw_softmax"].shape)

            mcd_softmax_dist = method_dict.get("raw_output_dist")
            if mcd_softmax_dist is not None:
                try:
                    mcd_softmax_dist = mcd_softmax_dist.f.arr_0
                except:
                    pass
                mcd_softmax_mean = np.mean(mcd_softmax_dist, axis=2)
                mcd_correct = (np.argmax(mcd_softmax_mean, axis=1) == method_dict["raw_labels"]) * 1
                method_dict["raw_mcd_correct"] = mcd_correct
                method_dict["raw_mcd_softmax_mean"] = mcd_softmax_mean
                method_dict["raw_mcd_softmax_dist"] = mcd_softmax_dist

            raw_external_confids = method_dict.get("raw_external_confids")
            if raw_external_confids is not None:
                try:
                    raw_external_confids = raw_external_confids.f.arr_0
                except:
                    pass
                method_dict["raw_external_confids"] = raw_external_confids
            raw_external_confids_dist = method_dict.get("raw_external_confids_dist")
            if raw_external_confids_dist is not None:
                try:
                    raw_external_confids_dist = raw_external_confids_dist.f.arr_0
                except:
                    pass
                method_dict["raw_external_confids_dist"] = raw_external_confids_dist



    def register_and_perform_studies(self):


        self.process_outputs()
        if self.qual_plot_confid:
            self.get_dataloader()
        flat_test_set_list = []
        for k, v in self.query_studies.items():
            print("QUERY STUDIES", self.query_studies)
            print(k, v, isinstance(v, list), type(v))
            if isinstance(v, list) or isinstance(v, omegaconf.listconfig.ListConfig):
                flat_test_set_list.extend([dataset for dataset in v])
            else:
                flat_test_set_list.append(v)

        print("CHECK flat list of all test datasets", flat_test_set_list)

        val_ix_shift = 1 if self.add_val_tuning else 0


        if self.add_val_tuning:

        # todo val tuning threhold study here.
        # get code from devries. outputs an optimal threshold which will later be used to compute extra metrics in all other studies.
            iid_set_ix = 0
            self.current_dataloader_ix = 0
            for method_dict in self.input_list:

                select_ix = np.argwhere(method_dict["raw_dataset_ix"] == iid_set_ix)[:, 0]

                method_dict["study_softmax"] = deepcopy(method_dict["raw_softmax"][select_ix])
                method_dict["study_labels"] = deepcopy(method_dict["raw_labels"][select_ix])
                method_dict["study_correct"] = deepcopy(method_dict["raw_correct"][select_ix])

                if method_dict.get("raw_external_confids") is not None:
                    method_dict["study_external_confids"] = deepcopy(method_dict["raw_external_confids"][select_ix])
                if method_dict.get("raw_external_confids_dist") is not None:
                    method_dict["study_external_confids_dist"] = deepcopy(method_dict["raw_external_confids_dist"][select_ix])
                    std = method_dict["study_external_confids_dist"].std(1)
                    print("CHECK EXT DIST", method_dict["study_external_confids_dist"].shape, std.mean(), std.min(), std.max(), method_dict["study_external_confids_dist"][0])

                if method_dict.get("raw_mcd_softmax_dist") is not None:
                    method_dict["study_mcd_softmax_mean"] = deepcopy(method_dict["raw_mcd_softmax_mean"][select_ix])
                    method_dict["study_mcd_softmax_dist"] = deepcopy(method_dict["raw_mcd_softmax_dist"][select_ix])
                    method_dict["study_mcd_correct"] = deepcopy(method_dict["raw_mcd_correct"][select_ix])

                self.rstar = method_dict["cfg"].eval.r_star
                self.rdelta = method_dict["cfg"].eval.r_delta
            self.perform_study("val_tuning")

        for study_name in self.query_studies.keys():

            if study_name == "iid_study":

                iid_set_ix = flat_test_set_list.index(self.query_studies[study_name])
                self.current_dataloader_ix = iid_set_ix + val_ix_shift
                for method_dict in self.input_list:

                    select_ix = np.argwhere(method_dict["raw_dataset_ix"] == iid_set_ix + val_ix_shift)[:, 0]

                    method_dict["study_softmax"] = deepcopy(method_dict["raw_softmax"][select_ix])
                    method_dict["study_labels"] = deepcopy(method_dict["raw_labels"][select_ix])
                    method_dict["study_correct"] = deepcopy(method_dict["raw_correct"][select_ix])

                    if method_dict.get("raw_external_confids") is not None:
                        method_dict["study_external_confids"] = deepcopy(method_dict["raw_external_confids"][select_ix])
                    if method_dict.get("raw_external_confids_dist") is not None:
                        method_dict["study_external_confids_dist"] = deepcopy(method_dict["raw_external_confids_dist"][select_ix])
                        std = method_dict["study_external_confids_dist"].std(1)
                        print("CHECK EXT DIST", method_dict["study_external_confids_dist"].shape, std.mean(), std.min(), std.max(), method_dict["study_external_confids_dist"][0])

                    if method_dict.get("raw_mcd_softmax_dist") is not None:
                        method_dict["study_mcd_softmax_mean"] = deepcopy(method_dict["raw_mcd_softmax_mean"][select_ix])
                        method_dict["study_mcd_softmax_dist"] = deepcopy(method_dict["raw_mcd_softmax_dist"][select_ix])
                        method_dict["study_mcd_correct"] = deepcopy(method_dict["raw_mcd_correct"][select_ix])

                self.perform_study(study_name)

            if study_name == "in_class_study":


                for method_dict in self.input_list:
                    for in_class_set in self.query_studies[study_name]:
                        iid_set_ix = flat_test_set_list.index(in_class_set) + val_ix_shift
                        self.current_dataloader_ix = iid_set_ix
                        select_ix = np.argwhere(method_dict["raw_dataset_ix"] == iid_set_ix)[:, 0]

                        method_dict["study_softmax"] = deepcopy(method_dict["raw_softmax"][select_ix])
                        method_dict["study_labels"] = deepcopy(method_dict["raw_labels"][select_ix])
                        method_dict["study_correct"] = deepcopy(method_dict["raw_correct"][select_ix])

                        if method_dict.get("raw_external_confids") is not None:
                            method_dict["study_external_confids"] = deepcopy(method_dict["raw_external_confids"][select_ix])
                        if method_dict.get("raw_external_confids_dist") is not None:
                            method_dict["study_external_confids_dist"] = deepcopy(method_dict["raw_external_confids_dist"][select_ix])
                            std = method_dict["study_external_confids_dist"].std(1)
                            print("CHECK EXT DIST", method_dict["study_external_confids_dist"].shape, std.mean(), std.min(), std.max(), method_dict["study_external_confids_dist"][0])

                        if method_dict.get("raw_mcd_softmax_dist") is not None:
                            method_dict["study_mcd_softmax_mean"] = deepcopy(method_dict["raw_mcd_softmax_mean"][select_ix])
                            method_dict["study_mcd_softmax_dist"] = deepcopy(method_dict["raw_mcd_softmax_dist"][select_ix])
                            method_dict["study_mcd_correct"] = deepcopy(method_dict["raw_mcd_correct"][select_ix])

                        self.perform_study("in_class_study_{}".format(in_class_set))


            if study_name == "new_class_study":


                for method_dict in self.input_list:

                    for new_class_set in self.query_studies[study_name]:
                        for mode in ["original_mode", "proposed_mode"]:

                            iid_set_ix = flat_test_set_list.index(self.query_studies["iid_study"]) + val_ix_shift
                            new_class_set_ix = flat_test_set_list.index(new_class_set) + val_ix_shift
                            self.current_dataloader_ix = new_class_set_ix
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
                            if method_dict.get("raw_external_confids") is not None:
                                method_dict["study_external_confids"] = deepcopy(method_dict["raw_external_confids"][select_ix_all])

                            if method_dict.get("raw_mcd_softmax_dist") is not None:
                                correct = deepcopy(method_dict["raw_mcd_correct"])
                                correct[select_ix_out] = 0
                                if mode == "original_mode":
                                    correct[select_ix_in] = 1

                                select_ix_all = np.argwhere((method_dict["raw_dataset_ix"] == new_class_set_ix) | ((method_dict["raw_dataset_ix"] == iid_set_ix) & (correct == 1)))[:, 0]
                                method_dict["study_mcd_softmax_mean"] = deepcopy(method_dict["raw_mcd_softmax_mean"][select_ix_all])
                                method_dict["study_mcd_softmax_dist"] = deepcopy(method_dict["raw_mcd_softmax_dist"][select_ix_all])
                                method_dict["study_mcd_correct"] = correct[select_ix_all]
                                if method_dict.get("raw_external_confids_dist") is not None:
                                    method_dict["study_external_confids_dist"] = deepcopy(
                                        method_dict["raw_external_confids_dist"][select_ix_all])

                            self.perform_study(study_name="{}_{}_{}".format(study_name, new_class_set, mode))


            if study_name == "noise_study":

                for noise_set in self.query_studies[study_name]:


                    # here could be a loop over explicit corruptions.
                    for intensity_level in range(5):

                        for method_dict in self.input_list:

                            noise_set_ix = flat_test_set_list.index(noise_set) + val_ix_shift
                            self.current_dataloader_ix = noise_set_ix

                            select_ix = np.argwhere(method_dict["raw_dataset_ix"] == noise_set_ix)[:, 0]

                            # new shape: n_corruptions, n_intensity_levels, n_test_cases, n_classes
                            method_dict["study_softmax"] = deepcopy(method_dict["raw_softmax"][select_ix]).reshape(
                                15, 5, -1, method_dict["raw_softmax"].shape[-1])[:, intensity_level].reshape(
                                -1, method_dict["raw_softmax"].shape[-1])
                            method_dict["study_labels"] = deepcopy(method_dict["raw_labels"][select_ix]).reshape(
                                15, 5, -1)[:, intensity_level].reshape(-1)
                            method_dict["study_correct"] = deepcopy(method_dict["raw_correct"][select_ix]).reshape(
                                15, 5, -1)[:, intensity_level].reshape(-1)
                            self.dummy_noise_ixs = np.arange(len(method_dict["raw_correct"][select_ix])).reshape(
                                15, 5, -1)[:, intensity_level].reshape(-1)
                            if method_dict.get("raw_external_confids") is not None:
                                method_dict["study_external_confids"] = deepcopy(
                                    method_dict["raw_external_confids"][select_ix]).reshape(15, 5, -1)[:, intensity_level].reshape(-1)
                            if method_dict.get("raw_external_confids_dist") is not None:
                                # method_dict["study_external_confids_dist"] = deepcopy(
                                #     method_dict["raw_external_confids_dist"][select_ix]).reshape(15, 5, -1)[:, intensity_level].reshape(-1)

                                method_dict["study_external_confids_dist"] = deepcopy(
                                    method_dict["raw_external_confids_dist"][select_ix]).reshape(
                                    15, 5, -1, method_dict["raw_external_confids_dist"].shape[-1])[:, intensity_level].reshape(
                                    -1, method_dict["raw_external_confids_dist"].shape[-1])

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
            predict = np.argmax(softmax, axis=1)

            if any("mcd" in cfd for cfd in method_dict["query_confids"]):
                mcd_softmax_mean = method_dict["study_mcd_softmax_mean"]
                mcd_predict = np.argmax(mcd_softmax_mean, axis=1)
                mcd_softmax_dist = method_dict["study_mcd_softmax_dist"]
                mcd_correct = method_dict["study_mcd_correct"]
                mcd_performance_metrics = self.compute_performance_metrics(mcd_softmax_mean, labels, mcd_correct, method_dict)

            performance_metrics = self.compute_performance_metrics(softmax, labels, correct, method_dict)
            # here is where threshold considerations would come int
            # also with BDL methods here first the merging method needs to be decided.

            # The case that test measures are not in val is prohibited anyway, because mcd-softmax output needs to fit.
            if "det_mcp" in method_dict["query_confids"]:
                method_dict["det_mcp"] = {}
                method_dict["det_mcp"]["confids"] = np.max(softmax, axis=1)
                method_dict["det_mcp"]["correct"] = deepcopy(correct)
                method_dict["det_mcp"]["metrics"] = deepcopy(performance_metrics)
                method_dict["det_mcp"]["predict"] = deepcopy(predict)
                print("CHECK DET MCP", np.sum(method_dict["det_mcp"]["confids"]), np.sum(method_dict["det_mcp"]["correct"]))

            if "det_pe" in method_dict["query_confids"]:
                method_dict["det_pe"] = {}
                method_dict["det_pe"]["confids"] = np.sum(softmax * (- np.log(softmax + 1e-7)), axis=1)
                method_dict["det_pe"]["correct"] = deepcopy(correct)
                method_dict["det_pe"]["metrics"] = deepcopy(performance_metrics)
                method_dict["det_pe"]["predict"] = deepcopy(predict)

            if "mcd_mcp" in method_dict["query_confids"]:
                method_dict["mcd_mcp"] = {}
                tmp_confids = np.max(mcd_softmax_mean, axis=1)
                method_dict["mcd_mcp"]["confids"] = tmp_confids
                method_dict["mcd_mcp"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_mcp"]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict["mcd_mcp"]["predict"] = deepcopy(mcd_predict)

            if "mcd_pe" in method_dict["query_confids"]:
                method_dict["mcd_pe"] = {}
                tmp_confids = np.sum(mcd_softmax_mean *
                                     (- np.log(mcd_softmax_mean + 1e-7)), axis=1)
                method_dict["mcd_pe"]["confids"] = tmp_confids
                method_dict["mcd_pe"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_pe"]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict["mcd_pe"]["predict"] = deepcopy(mcd_predict)

            if "mcd_ee" in method_dict["query_confids"]:
                method_dict["mcd_ee"] = {}
                tmp_confids = np.mean(np.sum(mcd_softmax_dist *
                                             (- np.log(mcd_softmax_dist + 1e-7)), axis=1), axis=1)
                method_dict["mcd_ee"]["confids"] = tmp_confids
                method_dict["mcd_ee"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_ee"]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict["mcd_ee"]["predict"] = deepcopy(mcd_predict)

            if "mcd_mi" in method_dict["query_confids"]:
                method_dict["mcd_mi"] = {}
                tmp_confids = method_dict["mcd_pe"]["confids"]-method_dict["mcd_ee"]["confids"]
                method_dict["mcd_mi"]["confids"] = tmp_confids
                method_dict["mcd_mi"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_mi"]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict["mcd_mi"]["predict"] = deepcopy(mcd_predict)

            if "mcd_sv" in method_dict["query_confids"]:
                method_dict["mcd_sv"] = {}
                # [b, cl, mcd] - [b, cl]
                print("CHECK FINAL DIST SHAPE", mcd_softmax_dist.shape)
                tmp_confids = np.mean(np.std(mcd_softmax_dist, axis=2), axis=(1))
                # tmp_confids = np.mean((mcd_softmax_dist - np.expand_dims(mcd_softmax_mean, axis=2))**2, axis=(1,2))
                method_dict["mcd_sv"]["confids"] = tmp_confids
                method_dict["mcd_sv"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_sv"]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict["mcd_sv"]["predict"] = deepcopy(mcd_predict)

            if "mcd_waic" in method_dict["query_confids"]:
                method_dict["mcd_waic"] = {}
                # [b, cl, mcd] - [b, cl]
                tmp_confids = np.max(mcd_softmax_mean, axis=1) - np.take(np.std(mcd_softmax_dist, axis=2), np.argmax(mcd_softmax_mean, axis=1))
                method_dict["mcd_waic"]["confids"] = tmp_confids
                method_dict["mcd_waic"]["correct"] = deepcopy(mcd_correct)
                method_dict["mcd_waic"]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict["mcd_waic"]["predict"] = deepcopy(mcd_predict)

            if any(cfd in method_dict["query_confids"] for cfd  in ["ext_waic", "bpd_waic", "tcp_waic", "dg_waic", "devries_waic"]):
                ext_confid_name = method_dict["cfg"].eval.ext_confid_name
                out_name = ext_confid_name + "_waic"
                method_dict[out_name] = {}
                tmp_confids = np.mean(method_dict["study_external_confids_dist"], axis=1) - np.std(method_dict["study_external_confids_dist"], axis=1)
                method_dict[out_name]["confids"] = tmp_confids
                method_dict[out_name]["correct"] = deepcopy(mcd_correct)
                method_dict[out_name]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict[out_name]["predict"] = deepcopy(mcd_predict)
                method_dict["query_confids"] = [out_name  if v=="ext_waic" else v for v in method_dict["query_confids"]]

            if any(cfd in method_dict["query_confids"] for cfd  in ["ext_mcd", "bpd_mcd", "tcp_mcd", "dg_mcd", "devries_mcd"]):
                ext_confid_name = method_dict["cfg"].eval.ext_confid_name
                out_name = ext_confid_name + "_mcd"
                method_dict[out_name] = {}
                tmp_confids = np.mean(method_dict["study_external_confids_dist"], axis=1)
                method_dict[out_name]["confids"] = tmp_confids
                method_dict[out_name]["correct"] = deepcopy(mcd_correct)
                method_dict[out_name]["metrics"] = deepcopy(mcd_performance_metrics)
                method_dict[out_name]["predict"] = deepcopy(mcd_predict)
                method_dict["query_confids"] = [out_name if v=="ext_mcd" else v for v in method_dict["query_confids"]]

            if any(cfd in method_dict["query_confids"] for cfd  in ["ext", "bpd", "tcp", "dg", "devries"]):
                ext_confid_name = method_dict["cfg"].eval.ext_confid_name
                method_dict[ext_confid_name] = {}
                method_dict[ext_confid_name]["confids"] = method_dict["study_external_confids"]
                method_dict[ext_confid_name]["correct"] = deepcopy(correct)
                method_dict[ext_confid_name]["metrics"] = deepcopy(performance_metrics)
                method_dict[ext_confid_name]["predict"] = deepcopy(predict)
                method_dict["query_confids"] = [ext_confid_name  if v=="ext" else v for v in method_dict["query_confids"]]


    def compute_performance_metrics(self, softmax, labels, correct, method_dict):
        performance_metrics = {}
        num_classes = self.num_classes
        if "nll" in self.query_performance_metrics:
            if "new_class" in self.study_name:
                performance_metrics["nll"] = None
            else:
                y_one_hot = np.eye(num_classes)[labels.astype("int")]
                performance_metrics["nll"] = np.mean(- np.log(softmax + 1e-7) * y_one_hot)
        if "accuracy" in self.query_performance_metrics:
            performance_metrics["accuracy"] = np.sum(correct) / correct.size
        if "brier_score" in self.query_performance_metrics:
            if "new_class" in self.study_name:
                performance_metrics["brier_score"] = None
            else:
                y_one_hot = np.eye(num_classes)[labels.astype("int")] # [b, classes]
                mse = (softmax - y_one_hot) ** 2
                performance_metrics["brier_score"] = np.mean(np.sum(mse, axis=1))

        return performance_metrics

    def compute_confid_metrics(self):

        for ix, method_dict in enumerate(self.input_list):

            for confid_key in method_dict["query_confids"]:
                print(self.study_name, confid_key)
                confid_dict = method_dict[confid_key]
                if confid_key == "bpd":
                    print("CHECK BEFORE NORM VALUES CORRECT", np.median(confid_dict["confids"][confid_dict["correct"] == 1]))
                    print("CHECK BEFORE NORM VALUES INCORRECT", np.median(confid_dict["confids"][confid_dict["correct"] == 0]))
                if any(cfd in confid_key for cfd  in ["_pe", "_ee", "_mi", "_sv", "bpd"]):
                    unnomred_confids = confid_dict["confids"].astype(np.float64)
                    min_confid = np.min(unnomred_confids)
                    max_confid = np.max(unnomred_confids)
                    confid_dict["confids"] = 1 - ((unnomred_confids - min_confid) / (max_confid - min_confid + 1e-9))

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

                if self.study_name == "val_tuning":
                    self.val_risk_scores[confid_key] = eval.get_val_risk_scores(self.rstar, self.rdelta[0]) # dummy, because now doing the plot and delta is a list!
                if self.val_risk_scores.get(confid_key) is not None:
                    val_risk_scores = self.val_risk_scores[confid_key]
                    test_risk_scores = {}
                    selected_residuals = 1 - confid_dict["correct"][np.argwhere(confid_dict["confids"] > val_risk_scores["theta"])]
                    test_risk_scores["test_risk"] = (np.sum(selected_residuals) / (len(selected_residuals) + 1e-9))
                    test_risk_scores["test_cov"] = len(selected_residuals) / len(confid_dict["correct"])
                    test_risk_scores["diff_risk"] = test_risk_scores["test_risk"] - self.rstar
                    test_risk_scores["diff_cov"] = test_risk_scores["test_cov"] - val_risk_scores["val_cov"]
                    test_risk_scores["rstar"] = self.rstar
                    test_risk_scores["val_theta"] = val_risk_scores["theta"]
                    confid_dict["metrics"].update(test_risk_scores)
                    if "test_risk" not in self.query_confid_metrics:
                        self.query_confid_metrics.extend(["test_risk", "test_cov", "diff_risk", "diff_cov", "rstar", "val_theta"])

                print("checking in", self.threshold_plot_confid, confid_key)
                if self.threshold_plot_confid is not None and confid_key == self.threshold_plot_confid:
                    if self.study_name == "val_tuning":
                        eval = ConfidEvaluator(confids=confid_dict["confids"],
                                               correct=confid_dict["correct"],
                                               query_metrics=self.query_confid_metrics,
                                               query_plots=self.query_plots,
                                               bins=self.calibration_bins)
                        self.threshold_plot_dict = {}
                        self.plot_threshs = []
                        print("creating threshold_plot_dict....")
                        for delta in self.rdelta:
                            plot_val_risk_scores = eval.get_val_risk_scores(self.rstar, delta)
                            self.plot_threshs.append(plot_val_risk_scores["theta"])
                            print(self.rstar, delta, plot_val_risk_scores["theta"], plot_val_risk_scores["val_risk"])

                    plot_string = "r*: {:.2f} \n".format(self.rstar)
                    for ix, thresh in enumerate(self.plot_threshs):
                        selected_residuals = 1 - confid_dict["correct"][
                            np.argwhere(confid_dict["confids"] > thresh)]
                        emp_risk = (np.sum(selected_residuals) / (len(selected_residuals) + 1e-9))
                        emp_coverage = len(selected_residuals) / len(confid_dict["correct"])
                        diff_risk = emp_risk - self.rstar
                        plot_string += "delta: {:.3f}: ".format(self.rdelta[ix])
                        plot_string += "thresh: {:.3f}: ".format(thresh)
                        plot_string += "emp.risk: {:.3f} ".format(emp_risk)
                        plot_string += "diff risk: {:.3f} ".format(diff_risk)
                        plot_string += "emp.cov.: {:.3f} \n".format(emp_coverage)

                    eval = ConfidEvaluator(confids=confid_dict["confids"],
                                           correct=confid_dict["correct"],
                                           query_metrics=self.query_confid_metrics,
                                           query_plots=self.query_plots,
                                           bins=self.calibration_bins)
                    true_thresh = eval.get_val_risk_scores(self.rstar, 0.1, no_bound_mode=True)["theta"]

                    print("creating new dict entry", self.study_name)
                    self.threshold_plot_dict[self.study_name] = {}
                    self.threshold_plot_dict[self.study_name]["confids"] =  confid_dict["confids"]
                    self.threshold_plot_dict[self.study_name]["correct"] = confid_dict["correct"]
                    self.threshold_plot_dict[self.study_name]["plot_string"] = plot_string
                    self.threshold_plot_dict[self.study_name]["true_thresh"] = true_thresh
                    self.threshold_plot_dict[self.study_name]["delta_threshs"] = self.plot_threshs
                    self.threshold_plot_dict[self.study_name]["deltas"] = self.rdelta

                if self.qual_plot_confid is not None and confid_key == self.qual_plot_confid:
                    top_k = 3

                    dataset =  self.test_dataloaders[self.current_dataloader_ix].dataset
                    if hasattr(dataset, "imgs"):
                        dataset_len = len(dataset.imgs)
                    elif hasattr(dataset, "data"):
                        dataset_len = len(dataset.data)
                    elif hasattr(dataset, "__len__"):
                        dataset_len = len(dataset.__len__)

                    if "new_class" in self.study_name:
                        keys = ["confids", "correct", "predict"]
                        for k in keys:
                            confid_dict[k] = confid_dict[k][-dataset_len:]
                    if not "noise" in self.study_name:
                        assert len(confid_dict["correct"]) == dataset_len
                    else:
                        assert len(confid_dict["correct"]) * 5 == dataset_len  == len(self.dummy_noise_ixs) * 5

                    # FP: high confidence, wrong correction, top-k parameter
                    incorrect_ixs = np.argwhere(confid_dict["correct"] == 0)[:, 0]
                    selected_confs = confid_dict["confids"][incorrect_ixs]
                    sorted_confs = np.argsort(selected_confs)[::-1][:top_k]  #flip ascending
                    fp_ixs = incorrect_ixs[sorted_confs]

                    fp_dict = {}
                    fp_dict["images"] = []
                    fp_dict["labels"] = []
                    fp_dict["predicts"] = []
                    fp_dict["confids"] = []
                    for ix in fp_ixs:
                        fp_dict["predicts"].append(confid_dict["predict"][ix])
                        fp_dict["confids"].append(confid_dict["confids"][ix])
                        if "noise" in self.study_name:
                            ix = self.dummy_noise_ixs[ix]
                        img, label = dataset[ix]
                        fp_dict["images"].append(img)
                        fp_dict["labels"].append(label)
                    # FN: low confidence, correct prediction, top-k parameter
                    correct_ixs = np.argwhere(confid_dict["correct"] == 1)[:, 0]
                    selected_confs = confid_dict["confids"][correct_ixs]
                    sorted_confs = np.argsort(selected_confs)[:top_k]  #keep ascending
                    fn_ixs = correct_ixs[sorted_confs]

                    fn_dict = {}
                    fn_dict["images"] = []
                    fn_dict["labels"] = []
                    fn_dict["predicts"] = []
                    fn_dict["confids"] = []
                    if not "new_class" in self.study_name:
                        for ix in fn_ixs:
                            fn_dict["predicts"].append(confid_dict["predict"][ix])
                            fn_dict["confids"].append(confid_dict["confids"][ix])
                            if "noise" in self.study_name:
                                ix = self.dummy_noise_ixs[ix]
                            img, label = dataset[ix]
                            fn_dict["images"].append(img)
                            fn_dict["labels"].append(label)


                    if hasattr(dataset, "classes") and "tinyimagenet" not in self.study_name:
                        fp_dict["labels"] = [dataset.classes[l] for l in fp_dict["labels"]]
                        if not "new_class" in self.study_name:
                            fp_dict["predicts"] = [dataset.classes[l] for l in fp_dict["predicts"]]
                        else:
                            fp_dict["predicts"] = [cifar100_classes[l] for l in fp_dict["predicts"]]
                        fn_dict["labels"] = [dataset.classes[l] for l in fn_dict["labels"]]
                        fn_dict["predicts"] = [dataset.classes[l] for l in fn_dict["predicts"]]
                    elif "new_class" in self.study_name:
                        fp_dict["predicts"] = [cifar100_classes[l] for l in fp_dict["predicts"]]


                    if "noise" in self.study_name:
                        for ix in fn_ixs:
                            corr_ix = self.dummy_noise_ixs[ix] % 50000
                            corr_ix = corr_ix // 10000
                            print("noise sanity check", corr_ix, self.dummy_noise_ixs[ix])

                    out_path = os.path.join(self.analysis_out_dir,
                                            "qual_plot_{}_{}.png".format(self.qual_plot_confid, self.study_name))
                    qual_plot(fp_dict, fn_dict, out_path)




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
                               method_dict["study_mcd_softmax_mean"].shape[0] if "mcd" in confid_key else
                               method_dict["study_softmax"].shape[0]]
                submit_list += [method_dict[confid_key]["metrics"][x] for x in all_metrics]
                df.loc[len(df)] = submit_list
        # print("CHECK SHIFT", self.study_name, all_metrics, self.input_list[0]["det_mcp"].keys())
        df.to_csv(os.path.join(self.analysis_out_dir, "analysis_metrics_{}.csv").format(self.study_name),
                  float_format='%.5f', decimal='.')
        print("saved csv to ", os.path.join(self.analysis_out_dir, "analysis_metrics_{}.csv".format(self.study_name)))

        group_file_path = os.path.join(self.input_list[0]["cfg"].exp.group_dir, "group_analysis_metrics.csv")
        if os.path.exists(group_file_path):
            with open(group_file_path, 'a') as f:
                df.to_csv(f, float_format='%.5f', decimal='.', header=False)
        else:
            with open(group_file_path, 'w') as f:
                df.to_csv(f, float_format='%.5f', decimal='.')



    def create_threshold_plot(self):
        # get overall with one dict per compared_method (i.e confid)
        f = ThresholdPlot(self.threshold_plot_dict)
        f.savefig(os.path.join(self.analysis_out_dir, "threshold_plot_{}.png".format(self.threshold_plot_confid)))
        print("saved threshold_plot to ", os.path.join(self.analysis_out_dir, "threshold_plot_{}.png".format(self.threshold_plot_confid)))


    def create_master_plot(self):
        # get overall with one dict per compared_method (i.e confid)
        input_dict = {"{}_{}".format(method_dict["name"], k):method_dict[k] for method_dict in self.input_list for k in method_dict["query_confids"] }
        plotter = ConfidPlotter(input_dict, self.query_plots, self.calibration_bins, fig_scale=1) # fig_scale big: 5
        f = plotter.compose_plot()
        f.savefig(os.path.join(self.analysis_out_dir, "master_plot_{}.png".format(self.study_name)))
        print("saved masterplot to ", os.path.join(self.analysis_out_dir, "master_plot_{}.png".format(self.study_name)))


    def get_dataloader(self):
        from src.loaders.abstract_loader import AbstractDataLoader
        dm = AbstractDataLoader(self.input_list[0]["cfg"], no_norm_flag=True)
        dm.prepare_data()
        dm.setup()
        self.test_dataloaders = dm.test_dataloader()




# TODO MUTLIPLE METHOD DICTS IS BROKEN! THIS SCRIPT SHOULD ONLY PROCESS 1 METHOD DICT ANYWAY!
def main(in_path=None, out_path=None, query_studies=None, add_val_tuning=True, cf=None, threshold_plot_confid="mcd_mcp", qual_plot_confid=None): #qual plot to false

    # path to the dir where the raw otuputs lie. NO SLASH AT THE END!
    if in_path is None: # NO SLASH AT THE END OF PATH !
        path_to_test_dir_list = [
            # "/mnt/hdd2/checkpoints/checks/check_mnist/test_results",
            "/mnt/hdd2/checkpoints/analysis/breeds_dg_bbresnet50_do1_run1_rew6/test_results",
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


    query_performance_metrics = ['accuracy', 'nll', 'brier_score']
    query_confid_metrics = ['failauc',
                            'failap_suc',
                            'failap_err',
                            "mce",
                            "ece",
                            "e-aurc",
                            "aurc",
                            "fpr@95tpr",
                            "risk@100cov",
                            "risk@95cov",
                            "risk@90cov",
                            "risk@85cov",
                            "risk@80cov",
                            "risk@75cov",
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
                        analysis_out_dir=analysis_out_dir,
                        add_val_tuning=add_val_tuning,
                        threshold_plot_confid=threshold_plot_confid,
                        qual_plot_confid=qual_plot_confid,
                        cf=cf
                        )

    analysis.register_and_perform_studies()
    if threshold_plot_confid is not None:
        analysis.create_threshold_plot()



if __name__ == '__main__':
   main()