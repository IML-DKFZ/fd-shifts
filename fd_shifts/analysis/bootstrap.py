from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.utils import resample

from fd_shifts import configs
from fd_shifts.analysis import Analysis, ExperimentData

from .studies import get_study_iterator

RANDOM_SEED = 10


def bootstrap_openset_data_iterator(analysis: AnalysisBS):
    raise NotImplementedError()


def bootstrap_new_class_data_iterator(
    data: ExperimentData,
    iid_set_name,
    dataset_name,
    n_bs: int,
    bs_size: int,
    stratified: bool = False,
):
    assert data.correct is not None
    iid_set_ix = data.dataset_name_to_idx(iid_set_name)
    new_class_set_ix = data.dataset_name_to_idx(dataset_name)

    select_ix_out = np.argwhere(data.dataset_idx == new_class_set_ix)[:, 0]

    correct = deepcopy(data.correct)
    correct[select_ix_out] = 0
    labels = deepcopy(data.labels)
    labels[select_ix_out] = -99

    # Select the two datasets
    select_ix_all = np.argwhere(
        (data.dataset_idx == new_class_set_ix) | (data.dataset_idx == iid_set_ix)
    )[:, 0]

    # Create bootstrap indices. By default, do stratification w.r.t. dataset. If
    # stratified==True, it is done w.r.t. failure label.
    n = len(select_ix_all)
    bs_indices = np.vstack(
        [
            resample(
                select_ix_all,
                n_samples=n if bs_size is None else bs_size,
                stratify=(
                    correct[select_ix_all]
                    if stratified
                    else data.dataset_idx[select_ix_all]
                ),
                random_state=rs + RANDOM_SEED,
            )
            for rs in range(n_bs)
        ]
    )

    mcd_correct = deepcopy(data.mcd_correct)
    select_ix_all_mcd = None
    if mcd_correct is not None:
        mcd_correct[select_ix_out] = 0
        select_ix_all_mcd = np.argwhere(
            (data.dataset_idx == new_class_set_ix) | (data.dataset_idx == iid_set_ix)
        )[:, 0]

        n = len(select_ix_all_mcd)
        bs_indices_mcd = np.vstack(
            [
                resample(
                    select_ix_all_mcd,
                    n_samples=n,
                    stratify=(
                        mcd_correct[select_ix_all_mcd]
                        if stratified
                        else data.dataset_idx[select_ix_all_mcd]
                    ),
                    random_state=rs + RANDOM_SEED,
                )
                for rs in range(n_bs)
            ]
        )
    else:
        bs_indices_mcd = n_bs * [None]

    def __filter_if_exists(data: npt.NDArray[Any] | None, mask):
        if data is not None:
            return data[mask]
        return None

    for bs_idx, (bs_selection, bs_selection_mcd) in enumerate(
        zip(bs_indices, bs_indices_mcd)
    ):
        # De-select incorrect inlier predictions
        bs_selection = bs_selection[
            (correct[bs_selection] == 1)
            | (data.dataset_idx[bs_selection] == new_class_set_ix)
        ]

        if bs_selection_mcd is not None:
            bs_selection_mcd = bs_selection_mcd[
                (mcd_correct[bs_selection_mcd] == 1)
                | (data.dataset_idx[bs_selection_mcd] == new_class_set_ix)
            ]

        yield bs_idx, data.__class__(
            softmax_output=data.softmax_output[bs_selection],
            logits=__filter_if_exists(data.logits, bs_selection),
            labels=labels[bs_selection],
            dataset_idx=data.dataset_idx[bs_selection],
            mcd_softmax_dist=__filter_if_exists(
                data.mcd_softmax_dist, bs_selection_mcd
            ),
            mcd_logits_dist=__filter_if_exists(data.mcd_logits_dist, bs_selection_mcd),
            external_confids=__filter_if_exists(data.external_confids, bs_selection),
            mcd_external_confids_dist=__filter_if_exists(
                data.mcd_external_confids_dist, bs_selection_mcd
            ),
            config=data.config,
            _correct=__filter_if_exists(correct, bs_selection),
            _mcd_correct=__filter_if_exists(mcd_correct, bs_selection_mcd),
            _mcd_labels=__filter_if_exists(labels, bs_selection_mcd),
            _react_logits=__filter_if_exists(data.react_logits, bs_selection),
            _maha_dist=__filter_if_exists(data.maha_dist, bs_selection),
            _vim_score=__filter_if_exists(data.vim_score, bs_selection),
            _dknn_dist=__filter_if_exists(data.dknn_dist, bs_selection),
            _train_features=data._train_features,
        )


def bootstrap_iterator(
    data: ExperimentData, n_bs: int, bs_size: int, stratified: bool = False
):
    n = len(data.labels)
    bs_indices = np.vstack(
        [
            resample(
                np.arange(n),
                n_samples=n if bs_size is None else bs_size,
                stratify=data.correct if stratified else None,
                random_state=rs + RANDOM_SEED,
            )
            for rs in range(n_bs)
        ]
    )

    def __filter_if_exists(data: npt.NDArray[Any] | None, mask):
        if data is not None:
            return data[mask]
        return None

    for bs_idx, bs_selection in enumerate(bs_indices):
        yield bs_idx, data.__class__(
            softmax_output=data.softmax_output[bs_selection],
            logits=__filter_if_exists(data.logits, bs_selection),
            labels=data.labels[bs_selection],
            dataset_idx=data.dataset_idx[bs_selection],
            mcd_softmax_dist=__filter_if_exists(data.mcd_softmax_dist, bs_selection),
            mcd_logits_dist=__filter_if_exists(data.mcd_logits_dist, bs_selection),
            external_confids=__filter_if_exists(data.external_confids, bs_selection),
            mcd_external_confids_dist=__filter_if_exists(
                data.mcd_external_confids_dist, bs_selection
            ),
            config=data.config,
            _correct=__filter_if_exists(data.correct, bs_selection),
            _mcd_correct=__filter_if_exists(data.mcd_correct, bs_selection),
            _mcd_labels=__filter_if_exists(data.labels, bs_selection),
            _react_logits=__filter_if_exists(data.react_logits, bs_selection),
            _maha_dist=__filter_if_exists(data.maha_dist, bs_selection),
            _vim_score=__filter_if_exists(data.vim_score, bs_selection),
            _dknn_dist=__filter_if_exists(data.dknn_dist, bs_selection),
            _train_features=data._train_features,
        )


class AnalysisBS(Analysis):
    """Analysis wrapper function for bootstrap analysis"""

    def __init__(self, *args, n_bs: int, no_iid: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bs = n_bs
        self._create_bs_indices_only = False
        self.no_iid = no_iid
        self._skip = False

    def register_and_perform_studies(self, bs_size: int = None):
        """"""
        if self._skip:
            logger.info(
                f"SKIPPING BS analysis for {self.cfg.exp.dir}, external config already evaluated!"
            )
            return

        if self.add_val_tuning:
            self.rstar = self.cfg.eval.r_star
            self.rdelta = self.cfg.eval.r_delta
            for study_name, study_data in get_study_iterator("val_tuning")(
                "val_tuning", self
            ):
                self.study_name = study_name
                logger.info(f"Performing bootstrap study {self.study_name}")

                # For val_tuning, only do a single evaluation on the original data (no
                # bootstrapping)
                self._perform_bootstrap_study(0, study_data)

                csv_path = (
                    self.analysis_out_dir / f"analysis_metrics_{self.study_name}.csv"
                )
                logger.info(f"Saved csv to {csv_path}")

        if self.holdout_classes is not None:
            self.study_name = "openset_proposed_mode"
            for bs_idx, data in bootstrap_openset_data_iterator(self):
                self._perform_bootstrap_study(bs_idx, data)

            csv_path = self.analysis_out_dir / f"analysis_metrics_{self.study_name}.csv"
            logger.info(f"Saved csv to {csv_path}")
            return

        for query_study, _ in self.query_studies:
            if query_study == "new_class_study":
                for new_class in self.query_studies.new_class_study:
                    self.study_name = f"new_class_study_{new_class}_proposed_mode"
                    logger.info(f"Performing bootstrap study {self.study_name}")

                    for bs_idx, data in bootstrap_new_class_data_iterator(
                        self.experiment_data,
                        self.query_studies.iid_study,
                        new_class,
                        self.n_bs,
                        bs_size,
                    ):
                        self._perform_bootstrap_study(bs_idx, data)

                    csv_path = (
                        self.analysis_out_dir
                        / f"analysis_metrics_{self.study_name}.csv"
                    )
                    logger.info(f"Saved csv to {csv_path}")

            elif self.no_iid and query_study == "iid_study":
                logger.info("Skipping IID study.")
                continue
            else:
                for study_name, study_data in get_study_iterator(query_study)(
                    query_study, self
                ):
                    self.study_name = study_name
                    logger.info(f"Performing bootstrap study {self.study_name}")

                    for bs_idx, data in bootstrap_iterator(
                        study_data, self.n_bs, bs_size
                    ):
                        self._perform_bootstrap_study(bs_idx, data)

                    csv_path = (
                        self.analysis_out_dir
                        / f"analysis_metrics_{self.study_name}.csv"
                    )
                    logger.info(f"Saved csv to {csv_path}")

    def _perform_bootstrap_study(self, bs_idx: int, selected_data: ExperimentData):
        self._get_confidence_scores(selected_data)
        self._compute_confid_metrics()
        self._create_results_csv(selected_data, bs_idx)

    def _create_results_csv(self, study_data: ExperimentData, bs_index: int):
        """Creates/Overwrites the csv for bs_index == 0, otherwise appends to the csv."""
        all_metrics = self.query_performance_metrics + self.query_confid_metrics
        columns = [
            "name",
            "study",
            "model",
            "network",
            "fold",
            "confid",
            "n_test",
            "bootstrap_index",
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
                (
                    study_data.mcd_softmax_mean.shape[0]
                    if "mcd" in confid_key
                    else study_data.softmax_output.shape[0]
                ),
                bs_index,
            ]
            submit_list += [
                self.method_dict[confid_key]["metrics"][x] for x in all_metrics
            ]
            df.loc[len(df)] = submit_list

        create_new_file = bs_index == 0
        df.to_csv(
            os.path.join(self.analysis_out_dir, "analysis_metrics_{}.csv").format(
                self.study_name
            ),
            float_format="%.5f",
            decimal=".",
            mode="w" if create_new_file else "a",
            header=create_new_file,
        )


def run_bs_analysis(
    config: configs.Config,
    n_bs: int = 500,
    iid_only: bool = False,
    no_iid: bool = False,
    exclude_noise_study: bool = False,
):
    """Bootstrap analysis

    Args:
        config (configs.Config): Complete Configuration
        n_bs (int, optional): Number of bootstrap samples. Defaults to 500.
    """
    path_to_test_dir = config.test.dir
    analysis_out_dir = config.exp.output_paths.analysis / "bootstrap"
    analysis_out_dir.mkdir(exist_ok=True, parents=True)
    query_studies = config.eval.query_studies

    if iid_only:
        query_studies.noise_study.dataset = None
        query_studies.in_class_study = []
        query_studies.new_class_study = []

    if exclude_noise_study:
        query_studies.noise_study.dataset = None

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
        "augrc",
        # "augrc-CI95-l",
        # "augrc-CI95-h",
        # "augrc-CI95",
        "e-augrc",
        "augrc-ba",
        "aurc-ba",
        "fpr@95tpr",
        "risk@100cov",
        "risk@95cov",
        "risk@90cov",
        "risk@85cov",
        "risk@80cov",
        "risk@75cov",
    ]

    query_plots = []

    logger.info(
        "Starting bootstrap analysis with in_path {}, out_path {}, and query studies {}".format(
            path_to_test_dir, analysis_out_dir, query_studies
        )
    )

    bs_analysis = AnalysisBS(
        path=path_to_test_dir,
        query_performance_metrics=query_performance_metrics,
        query_confid_metrics=query_confid_metrics,
        query_plots=query_plots,
        query_studies=query_studies,
        analysis_out_dir=analysis_out_dir,
        add_val_tuning=config.eval.val_tuning,
        threshold_plot_confid=None,
        qual_plot_confid=None,
        cf=config,
        n_bs=n_bs,
        no_iid=no_iid,
    )

    bs_analysis.register_and_perform_studies()
