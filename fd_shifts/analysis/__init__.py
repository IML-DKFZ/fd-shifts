from __future__ import annotations

import os
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import inspect
from scipy import special as scpspecial
from sklearn.calibration import _sigmoid_calibration as calib

from fd_shifts import configs

from . import metrics
from .confid_scores import ConfidScore, SecondaryConfidScore, is_external_confid
from .eval_utils import (
    ConfidEvaluator,
    ConfidPlotter,
    ThresholdPlot,
    cifar100_classes,
    qual_plot,
)
from .studies import get_study_iterator


@dataclass
class ExperimentData:
    """Wrapper class containing the complete outputs of one experiment"""

    softmax_output: npt.NDArray[Any]
    labels: npt.NDArray[Any]
    dataset_idx: npt.NDArray[Any]

    config: configs.Config

    logits: npt.NDArray[Any] | None = None

    mcd_softmax_dist: npt.NDArray[Any] | None = None
    mcd_logits_dist: npt.NDArray[Any] | None = None

    external_confids: npt.NDArray[Any] | None = None
    mcd_external_confids_dist: npt.NDArray[Any] | None = None

    _mcd_correct: npt.NDArray[Any] | None = field(default=None)
    _mcd_labels: npt.NDArray[Any] | None = field(default=None)
    _correct: npt.NDArray[Any] | None = field(default=None)

    @property
    def correct(self) -> npt.NDArray[Any]:
        if self._correct is not None:
            return self._correct
        return (np.argmax(self.softmax_output, axis=1) == self.labels).astype(int)

    @property
    def mcd_softmax_mean(self) -> npt.NDArray[Any] | None:
        if self.mcd_softmax_dist is None:
            return None
        return np.mean(self.mcd_softmax_dist, axis=2)

    @property
    def mcd_logits_mean(self) -> npt.NDArray[Any] | None:
        if self.mcd_logits_dist is None:
            return None
        return np.mean(self.mcd_logits_dist, axis=2)

    @property
    def mcd_correct(self) -> npt.NDArray[Any] | None:
        if self._mcd_correct is not None:
            return self._mcd_correct
        if self.mcd_softmax_mean is None:
            return None
        return (np.argmax(self.mcd_softmax_mean, axis=1) == self.labels).astype(int)

    @property
    def mcd_labels(self) -> npt.NDArray[Any] | None:
        if self._mcd_labels is not None:
            return self._mcd_labels
        if self.mcd_softmax_mean is None:
            return None
        return self.labels

    def dataset_name_to_idx(self, dataset_name: str) -> int:
        if dataset_name == "val_tuning":
            return 0

        flat_test_set_list = []
        for _, datasets in self.config.eval.query_studies:
            if isinstance(datasets, (list, ListConfig)):
                flat_test_set_list.extend(list(datasets))
            else:
                flat_test_set_list.append(datasets)

        dataset_idx = flat_test_set_list.index(dataset_name)

        if self.config.eval.val_tuning:
            dataset_idx += 1

        return dataset_idx

    def filter_dataset_by_name(self, dataset_name: str) -> ExperimentData:
        return self.filter_dataset_by_index(self.dataset_name_to_idx(dataset_name))

    def filter_dataset_by_index(self, dataset_idx: int) -> ExperimentData:
        mask = np.argwhere(self.dataset_idx == dataset_idx)[:, 0]

        def _filter_if_exists(data: npt.NDArray[Any] | None):
            if data is not None:
                return data[mask]

            return None

        return ExperimentData(
            softmax_output=self.softmax_output[mask],
            logits=_filter_if_exists(self.logits),
            labels=self.labels[mask],
            dataset_idx=self.dataset_idx[mask],
            mcd_softmax_dist=_filter_if_exists(self.mcd_softmax_dist),
            mcd_logits_dist=_filter_if_exists(self.mcd_logits_dist),
            external_confids=_filter_if_exists(self.external_confids),
            mcd_external_confids_dist=_filter_if_exists(self.mcd_external_confids_dist),
            config=self.config,
        )

    @staticmethod
    def __load_npz_if_exists(path: Path) -> npt.NDArray[np.float64] | None:
        if not path.is_file():
            return None

        with np.load(path) as npz:
            return npz.f.arr_0

    @staticmethod
    def from_experiment(
        test_dir: Path,
        config: configs.Config,
        holdout_classes: list | None = None,
    ) -> ExperimentData:
        if not isinstance(test_dir, Path):
            test_dir = Path(test_dir)

        if (test_dir / "raw_logits.npz").is_file():
            with np.load(test_dir / "raw_logits.npz") as npz:
                raw_output = npz.f.arr_0

            logits = raw_output[:, :-2]
            softmax = scpspecial.softmax(logits, axis=1)

            if (
                mcd_logits_dist := ExperimentData.__load_npz_if_exists(
                    test_dir / "raw_logits_dist.npz"
                )
            ) is not None:
                mcd_softmax_dist = scpspecial.softmax(mcd_logits_dist, axis=1)
            else:
                mcd_softmax_dist = None

        elif (test_dir / "raw_output.npz").is_file():
            with np.load(test_dir / "raw_output.npz") as npz:
                raw_output = npz.f.arr_0

            logits = None
            mcd_logits_dist = None
            softmax = raw_output[:, :-2]
            mcd_softmax_dist = ExperimentData.__load_npz_if_exists(
                test_dir / "raw_output_dist.npz"
            )
        else:
            raise FileNotFoundError("Could not find model output")

        if holdout_classes is not None:
            softmax[:, holdout_classes] = 0

            if logits is not None:
                logits[:, holdout_classes] = -np.inf
                softmax = scpspecial.softmax(logits, axis=1)

            if mcd_softmax_dist is not None:
                mcd_softmax_dist[:, holdout_classes, :] = 0

            if mcd_logits_dist is not None:
                mcd_logits_dist[:, holdout_classes, :] = -np.inf
                mcd_softmax_dist = scpspecial.softmax(mcd_logits_dist, axis=1)

        external_confids = ExperimentData.__load_npz_if_exists(
            test_dir / "external_confids.npz"
        )
        mcd_external_confids_dist = ExperimentData.__load_npz_if_exists(
            test_dir / "external_confids_dist.npz"
        )

        return ExperimentData(
            softmax_output=softmax,
            logits=logits,
            labels=raw_output[:, -2],
            dataset_idx=raw_output[:, -1],
            mcd_softmax_dist=mcd_softmax_dist,
            mcd_logits_dist=mcd_logits_dist,
            external_confids=external_confids,
            mcd_external_confids_dist=mcd_external_confids_dist,
            config=config,
        )


@dataclass
class PlattScaling:
    """Platt scaling normalization function"""

    def __init__(self, val_confids: npt.NDArray[Any], val_correct: npt.NDArray[Any]):
        self.a: Number
        self.b: Number

        confids = val_confids[~np.isnan(val_confids)]
        correct = val_correct[~np.isnan(val_confids)]
        if len(confids) == 0:
            raise ValueError("All confids are NaN")
        self.a, self.b = calib(confids, correct)

    def __call__(self, confids: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return 1 / (1 + np.exp(confids * self.a + self.b))


@dataclass
class QuantileScaling:
    """Quantile scaling normalization function"""

    def __init__(self, val_confids: npt.NDArray[Any], quantile=0.01):
        self.quantile = np.quantile(val_confids, quantile)

    def __call__(self, confids: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return scpspecial.expit(-1 * (confids - self.quantile))


class Analysis:
    """Analysis wrapper function"""

    def __init__(
        self,
        path: Path,
        query_performance_metrics: list[str],
        query_confid_metrics: list[str],
        query_plots: list[str],
        query_studies: configs.QueryStudiesConfig,
        analysis_out_dir: Path,
        add_val_tuning: bool,
        threshold_plot_confid: str | None,
        qual_plot_confid,
        cf: configs.Config,
    ):
        self.method_dict = {"cfg": cf, "name": path.parts[-2]}

        self.cfg = cf

        self.holdout_classes: list | None = (
            kwargs.get("out_classes") if (kwargs := cf.data.kwargs) else None
        )
        self.experiment_data = ExperimentData.from_experiment(
            path, cf, self.holdout_classes
        )

        self.method_dict["query_confids"] = self.cfg.eval.confidence_measures.test
        if self.experiment_data.external_confids is None:
            self.method_dict["query_confids"] = list(
                filter(
                    lambda confid: "ext" not in confid,
                    self.method_dict["query_confids"],
                )
            )

        if self.experiment_data.mcd_softmax_dist is None:
            self.method_dict["query_confids"] = list(
                filter(
                    lambda confid: "mcd" not in confid,
                    self.method_dict["query_confids"],
                )
            )

        self.secondary_confids = []

        if (
            "ext" in self.method_dict["query_confids"]
            and self.cfg.eval.ext_confid_name == "maha"
        ):
            self.method_dict["query_confids"].append("ext_qt")
            self.secondary_confids.extend(
                [
                    "det_mcp-maha-average",
                    "det_mcp-maha-product",
                    "det_mcp-maha_qt-average",
                ]
            )
            self.method_dict["query_confids"].extend(self.secondary_confids)

        if self.experiment_data.logits is not None:
            self.method_dict["query_confids"].append("det_mls")

        if self.experiment_data.mcd_logits_dist is not None:
            self.method_dict["query_confids"].append("mcd_mls")

        logger.debug("CHECK QUERY CONFIDS\n{}", self.method_dict["query_confids"])

        self.query_performance_metrics = query_performance_metrics
        self.query_confid_metrics = query_confid_metrics
        self.query_plots = query_plots
        self.query_studies = (
            self.cfg.eval.query_studies if query_studies is None else query_studies
        )
        self.analysis_out_dir = analysis_out_dir
        self.calibration_bins = 20
        self.val_risk_scores = {}
        self.num_classes = self.cfg.data.num_classes
        self.add_val_tuning = add_val_tuning

        if not add_val_tuning:
            raise ValueError("Need val tuning to perform platt scaling")

        self.threshold_plot_confid = threshold_plot_confid
        self.qual_plot_confid = qual_plot_confid
        self.normalization_functions = {}

    def register_and_perform_studies(self):
        """Entry point to perform analysis defined in config"""

        if self.qual_plot_confid:
            self._get_dataloader()

        if self.add_val_tuning:
            self.rstar = self.cfg.eval.r_star
            self.rdelta = self.cfg.eval.r_delta
            for study_name, study_data in get_study_iterator("val_tuning")(
                "val_tuning", self
            ):
                self._perform_study("val_tuning", study_data)

        if self.holdout_classes is not None:
            for study_name, study_data in get_study_iterator("openset")(
                "openset", self
            ):
                self._perform_study(study_name, study_data)
            return

        for query_study, _ in self.query_studies:
            for study_name, study_data in get_study_iterator(query_study)(
                query_study, self
            ):
                self._perform_study(study_name, study_data)

    def _perform_study(self, study_name, study_data: ExperimentData):
        self.study_name = study_name
        self._get_confidence_scores(study_data)
        self._compute_confid_metrics()
        self._create_results_csv(study_data)

    def _fix_external_confid_name(self, name: str):
        if not is_external_confid(name):
            return name

        ext_confid_name = self.cfg.eval.ext_confid_name

        suffix = f"_{parts[1]}" if len(parts := name.split("_")) > 1 else ""

        query_confid = ext_confid_name + suffix

        self.method_dict["query_confids"] = [
            query_confid if v == name else v for v in self.method_dict["query_confids"]
        ]

        return query_confid

    def _get_confidence_scores(self, study_data: ExperimentData):
        for query_confid in self.method_dict["query_confids"]:
            if query_confid in self.secondary_confids:
                continue

            confid_score = ConfidScore(
                study_data=study_data,
                query_confid=query_confid,
                analysis=self,
            )

            query_confid = self._fix_external_confid_name(query_confid)

            confids = confid_score.confids

            if self.study_name == "val_tuning":
                if query_confid == "maha_qt":
                    self.normalization_functions[query_confid] = QuantileScaling(
                        confids
                    )
                elif query_confid == "temp_logits":
                    self.normalization_functions[query_confid] = TemperatureScaling(
                        confids, study_data.labels
                    )
                elif any(
                    cfd in query_confid
                    for cfd in ["_pe", "_ee", "_mi", "_sv", "bpd", "maha", "_mls"]
                ):
                    self.normalization_functions[query_confid] = PlattScaling(
                        confids, confid_score.correct
                    )
                else:
                    self.normalization_functions[query_confid] = lambda confids: confids

            assert not np.all(
                np.isnan(confids)
            ), f"Nan in {query_confid} in {self.study_name} before normalization"
            confids = self.normalization_functions[query_confid](confids)
            assert not np.all(
                np.isnan(confids)
            ), f"Nan in {query_confid} in {self.study_name} after normalization {inspect(self.normalization_functions[query_confid])}"

            self.method_dict[query_confid] = {}
            self.method_dict[query_confid]["confids"] = confids
            self.method_dict[query_confid]["labels"] = confid_score.labels
            self.method_dict[query_confid]["correct"] = confid_score.correct
            self.method_dict[query_confid]["metrics"] = confid_score.metrics
            self.method_dict[query_confid]["predict"] = confid_score.predict

        for query_confid in self.secondary_confids:
            confid_score = SecondaryConfidScore(
                study_data=study_data,
                query_confid=query_confid,
                analysis=self,
            )
            self.method_dict[query_confid] = {}
            self.method_dict[query_confid]["confids"] = confid_score.confids
            self.method_dict[query_confid]["correct"] = confid_score.correct
            self.method_dict[query_confid]["metrics"] = confid_score.metrics
            self.method_dict[query_confid]["predict"] = confid_score.predict
            self.method_dict[query_confid]["labels"] = confid_score.labels

    def _compute_performance_metrics(self, softmax, labels, correct):
        performance_metrics = {}
        num_classes = self.num_classes
        if "nll" in self.query_performance_metrics:
            if "new_class" in self.study_name or "openset" in self.study_name:
                performance_metrics["nll"] = None
            else:
                y_one_hot = np.eye(num_classes)[labels.astype("int")]
                performance_metrics["nll"] = np.mean(
                    -np.log(softmax + 1e-7) * y_one_hot
                )
        if "accuracy" in self.query_performance_metrics:
            performance_metrics["accuracy"] = np.sum(correct) / correct.size
        if "b-accuracy" in self.query_performance_metrics:
            accuracies_list = []
            for cla in np.unique(labels):
                is_class = labels == cla
                accuracy_class = np.mean(correct[is_class])
                accuracies_list.append(accuracy_class)
            performance_metrics["b-accuracy"] = np.mean(accuracies_list)
        if "brier_score" in self.query_performance_metrics:
            if "new_class" in self.study_name or "openset" in self.study_name:
                performance_metrics["brier_score"] = None
            else:
                y_one_hot = np.eye(num_classes)[labels.astype("int")]
                mse = (softmax - y_one_hot) ** 2
                performance_metrics["brier_score"] = np.mean(np.sum(mse, axis=1))

        return performance_metrics

    def _compute_confid_metrics(self):
        for confid_key in self.method_dict["query_confids"]:
            logger.debug("{}\n{}", self.study_name, confid_key)
            confid_dict = self.method_dict[confid_key]
            if confid_key == "bpd" or confid_key == "maha":
                logger.debug(
                    "CHECK BEFORE NORM VALUES CORRECT\n{}",
                    np.median(confid_dict["confids"][confid_dict["correct"] == 1]),
                )
                logger.debug(
                    "CHECK BEFORE NORM VALUES INCORRECT\n{}",
                    np.median(confid_dict["confids"][confid_dict["correct"] == 0]),
                )

            if confid_key == "bpd" or confid_key == "maha":
                logger.debug(
                    "CHECK AFTER NORM VALUES CORRECT\n{}",
                    np.median(confid_dict["confids"][confid_dict["correct"] == 1]),
                )
                logger.debug(
                    "CHECK AFTER NORM VALUES INCORRECT\n{}",
                    np.median(confid_dict["confids"][confid_dict["correct"] == 0]),
                )

            eval = ConfidEvaluator(
                confids=confid_dict["confids"],
                correct=confid_dict["correct"],
                labels=confid_dict.get("labels"),
                query_metrics=self.query_confid_metrics,
                query_plots=self.query_plots,
                bins=self.calibration_bins,
            )

            confid_dict["metrics"].update(eval.get_metrics_per_confid())
            confid_dict["plot_stats"] = eval.get_plot_stats_per_confid()

            if self.study_name == "val_tuning":
                self.val_risk_scores[confid_key] = eval.get_val_risk_scores(
                    self.rstar, self.rdelta
                )  # dummy, because now doing the plot and delta is a list!
            if self.val_risk_scores.get(confid_key) is not None:
                val_risk_scores = self.val_risk_scores[confid_key]
                test_risk_scores = {}
                selected_residuals = (
                    1
                    - confid_dict["correct"][
                        np.argwhere(confid_dict["confids"] > val_risk_scores["theta"])
                    ]
                )
                test_risk_scores["test_risk"] = np.sum(selected_residuals) / (
                    len(selected_residuals) + 1e-9
                )
                test_risk_scores["test_cov"] = len(selected_residuals) / len(
                    confid_dict["correct"]
                )
                test_risk_scores["diff_risk"] = (
                    test_risk_scores["test_risk"] - self.rstar
                )
                test_risk_scores["diff_cov"] = (
                    test_risk_scores["test_cov"] - val_risk_scores["val_cov"]
                )
                test_risk_scores["rstar"] = self.rstar
                test_risk_scores["val_theta"] = val_risk_scores["theta"]
                confid_dict["metrics"].update(test_risk_scores)
                if "test_risk" not in self.query_confid_metrics:
                    self.query_confid_metrics.extend(
                        [
                            "test_risk",
                            "test_cov",
                            "diff_risk",
                            "diff_cov",
                            "rstar",
                            "val_theta",
                        ]
                    )

            logger.debug("checking in\n{}\n{}", self.threshold_plot_confid, confid_key)
            if (
                self.threshold_plot_confid is not None
                and confid_key == self.threshold_plot_confid
            ):
                if self.study_name == "val_tuning":
                    eval = ConfidEvaluator(
                        confids=confid_dict["confids"],
                        correct=confid_dict["correct"],
                        query_metrics=self.query_confid_metrics,
                        query_plots=self.query_plots,
                        bins=self.calibration_bins,
                        labels=confid_dict["labels"],
                    )
                    self.threshold_plot_dict = {}
                    self.plot_threshs = []
                    self.true_covs = []
                    logger.debug("creating threshold_plot_dict....")
                    plot_val_risk_scores = eval.get_val_risk_scores(
                        self.rstar, self.rdelta
                    )
                    self.plot_threshs.append(plot_val_risk_scores["theta"])
                    self.true_covs.append(plot_val_risk_scores["val_cov"])
                    logger.debug(
                        "{}\n{}\n{}\n{}",
                        self.rstar,
                        self.rdelta,
                        plot_val_risk_scores["theta"],
                        plot_val_risk_scores["val_risk"],
                    )

                plot_string = "r*: {:.2f}  \n".format(self.rstar)
                for ix, thresh in enumerate(self.plot_threshs):
                    selected_residuals = (
                        1
                        - confid_dict["correct"][
                            np.argwhere(confid_dict["confids"] > thresh)
                        ]
                    )
                    emp_risk = np.sum(selected_residuals) / (
                        len(selected_residuals) + 1e-9
                    )
                    emp_coverage = len(selected_residuals) / len(confid_dict["correct"])
                    diff_risk = emp_risk - self.rstar
                    plot_string += "delta: {:.3f}: ".format(self.rdelta)
                    plot_string += "erisk: {:.3f} ".format(emp_risk)
                    plot_string += "diff risk: {:.3f} ".format(diff_risk)
                    plot_string += "ecov.: {:.3f} \n".format(emp_coverage)
                    plot_string += "diff cov.: {:.3f} \n".format(
                        emp_coverage - self.true_covs[ix]
                    )

                eval = ConfidEvaluator(
                    confids=confid_dict["confids"],
                    correct=confid_dict["correct"],
                    query_metrics=self.query_confid_metrics,
                    query_plots=self.query_plots,
                    bins=self.calibration_bins,
                    labels=confid_dict["labels"],
                )
                true_thresh = eval.get_val_risk_scores(
                    self.rstar, 0.1, no_bound_mode=True
                )["theta"]

                logger.debug("creating new dict entry\n{}", self.study_name)
                self.threshold_plot_dict[self.study_name] = {}
                self.threshold_plot_dict[self.study_name]["confids"] = confid_dict[
                    "confids"
                ]
                self.threshold_plot_dict[self.study_name]["correct"] = confid_dict[
                    "correct"
                ]
                self.threshold_plot_dict[self.study_name]["plot_string"] = plot_string
                self.threshold_plot_dict[self.study_name]["true_thresh"] = true_thresh
                self.threshold_plot_dict[self.study_name][
                    "delta_threshs"
                ] = self.plot_threshs
                self.threshold_plot_dict[self.study_name]["deltas"] = self.rdelta

            if (
                self.qual_plot_confid is not None
                and confid_key == self.qual_plot_confid
            ):
                top_k = 3

                dataset = self.test_dataloaders[self.current_dataloader_ix].dataset
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
                    assert (
                        len(confid_dict["correct"]) * 5
                        == dataset_len
                        == len(self.dummy_noise_ixs) * 5
                    )

                incorrect_ixs = np.argwhere(confid_dict["correct"] == 0)[:, 0]
                selected_confs = confid_dict["confids"][incorrect_ixs]
                sorted_confs = np.argsort(selected_confs)[::-1][:top_k]
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

                correct_ixs = np.argwhere(confid_dict["correct"] == 1)[:, 0]
                selected_confs = confid_dict["confids"][correct_ixs]
                sorted_confs = np.argsort(selected_confs)[:top_k]
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

                if (
                    hasattr(dataset, "classes")
                    and "tinyimagenet" not in self.study_name
                ):
                    fp_dict["labels"] = [dataset.classes[l] for l in fp_dict["labels"]]
                    if not "new_class" in self.study_name:
                        fp_dict["predicts"] = [
                            dataset.classes[l] for l in fp_dict["predicts"]
                        ]
                    else:
                        fp_dict["predicts"] = [
                            cifar100_classes[l] for l in fp_dict["predicts"]
                        ]
                    fn_dict["labels"] = [dataset.classes[l] for l in fn_dict["labels"]]
                    fn_dict["predicts"] = [
                        dataset.classes[l] for l in fn_dict["predicts"]
                    ]
                elif "new_class" in self.study_name:
                    fp_dict["predicts"] = [
                        cifar100_classes[l] for l in fp_dict["predicts"]
                    ]

                if "noise" in self.study_name:
                    for ix in fn_ixs:
                        corr_ix = self.dummy_noise_ixs[ix] % 50000
                        corr_ix = corr_ix // 10000
                        logger.debug(
                            "noise sanity check\n{}\n{}",
                            corr_ix,
                            self.dummy_noise_ixs[ix],
                        )

                out_path = os.path.join(
                    self.analysis_out_dir,
                    "qual_plot_{}_{}.png".format(
                        self.qual_plot_confid, self.study_name
                    ),
                )
                qual_plot(fp_dict, fn_dict, out_path)

    def _create_results_csv(self, study_data: ExperimentData):
        all_metrics = self.query_performance_metrics + self.query_confid_metrics
        columns = [
            "name",
            "study",
            "model",
            "network",
            "fold",
            "confid",
            "n_test",
        ] + all_metrics
        df = pd.DataFrame(columns=columns)
        network = self.cfg.model.network
        if network is not None:
            backbone = dict(network).get("backbone")
        else:
            backbone = None
        for confid_key in self.method_dict["query_confids"]:
            submit_list = [
                self.method_dict["name"],
                self.study_name,
                self.cfg.model.name,
                backbone,
                self.cfg.exp.fold,
                confid_key,
                study_data.mcd_softmax_mean.shape[0]
                if "mcd" in confid_key
                else study_data.softmax_output.shape[0],
            ]
            submit_list += [
                self.method_dict[confid_key]["metrics"][x] for x in all_metrics
            ]
            df.loc[len(df)] = submit_list
        df.to_csv(
            os.path.join(self.analysis_out_dir, "analysis_metrics_{}.csv").format(
                self.study_name
            ),
            float_format="%.5f",
            decimal=".",
        )
        logger.debug(
            "saved csv to {}",
            os.path.join(
                self.analysis_out_dir, "analysis_metrics_{}.csv".format(self.study_name)
            ),
        )

        group_file_path = os.path.join(
            self.cfg.exp.group_dir, "group_analysis_metrics.csv"
        )
        if os.path.exists(group_file_path):
            with open(group_file_path, "a") as f:
                df.to_csv(f, float_format="%.5f", decimal=".", header=False)
        else:
            with open(group_file_path, "w") as f:
                df.to_csv(f, float_format="%.5f", decimal=".")

    def _create_threshold_plot(self):
        f = ThresholdPlot(self.threshold_plot_dict)
        f.savefig(
            os.path.join(
                self.analysis_out_dir,
                "threshold_plot_{}.png".format(self.threshold_plot_confid),
            )
        )
        logger.debug(
            "saved threshold_plot to {}",
            os.path.join(
                self.analysis_out_dir,
                "threshold_plot_{}.png".format(self.threshold_plot_confid),
            ),
        )

    def _create_master_plot(self):
        input_dict = {
            "{}_{}".format(self.method_dict["name"], k): self.method_dict[k]
            for k in self.method_dict["query_confids"]
        }
        plotter = ConfidPlotter(
            input_dict, self.query_plots, self.calibration_bins, fig_scale=1
        )
        f = plotter.compose_plot()
        f.savefig(
            os.path.join(
                self.analysis_out_dir, "master_plot_{}.png".format(self.study_name)
            )
        )
        logger.debug(
            "saved masterplot to {}",
            os.path.join(
                self.analysis_out_dir, "master_plot_{}.png".format(self.study_name)
            ),
        )

    def _get_dataloader(self):
        from fd_shifts.loaders.data_loader import FDShiftsDataLoader

        dm = FDShiftsDataLoader(self.cfg, no_norm_flag=True)
        dm.prepare_data()
        dm.setup()
        self.test_dataloaders = dm.test_dataloader()


def main(
    in_path: Path,
    out_path: Path,
    cf: configs.Config,
    query_studies: configs.QueryStudiesConfig,
    add_val_tuning: bool = True,
    threshold_plot_confid: str | None = "tcp_mcd",
    qual_plot_confid=None,
):
    path_to_test_dir = in_path

    analysis_out_dir = out_path

    query_performance_metrics = ["accuracy", "b-accuracy", "nll", "brier_score"]
    query_confid_metrics = [
        "failauc",
        "failap_suc",
        "failap_err",
        "fail-NLL",
        "mce",
        "ece",
        "e-aurc",
        "b-aurc",
        "aurc",
        "fpr@95tpr",
        "risk@100cov",
        "risk@95cov",
        "risk@90cov",
        "risk@85cov",
        "risk@80cov",
        "risk@75cov",
    ]

    query_plots = []

    if not os.path.exists(analysis_out_dir):
        os.mkdir(analysis_out_dir)

    logger.debug(
        "starting analysis with in_path {}, out_path {}, and query studies {}".format(
            path_to_test_dir, analysis_out_dir, query_studies
        )
    )

    analysis = Analysis(
        path=path_to_test_dir,
        query_performance_metrics=query_performance_metrics,
        query_confid_metrics=query_confid_metrics,
        query_plots=query_plots,
        query_studies=query_studies,
        analysis_out_dir=analysis_out_dir,
        add_val_tuning=add_val_tuning,
        threshold_plot_confid=threshold_plot_confid,
        qual_plot_confid=qual_plot_confid,
        cf=cf,
    )

    analysis.register_and_perform_studies()
    if threshold_plot_confid is not None:
        analysis._create_threshold_plot()
