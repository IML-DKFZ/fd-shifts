from __future__ import annotations

import os
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Literal, overload

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from omegaconf import ListConfig
from rich import inspect
from scipy import special as scpspecial
from sklearn.calibration import _sigmoid_calibration as calib

from fd_shifts import configs

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

    _features: npt.NDArray[Any] | None = field(default=None)
    _train_features: npt.NDArray[Any] | None = field(default=None)
    _last_layer: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = field(default=None)

    _react_logits: npt.NDArray[Any] | None = field(default=None)
    _maha_dist: npt.NDArray[Any] | None = field(default=None)
    _vim_score: npt.NDArray[Any] | None = field(default=None)
    _dknn_dist: npt.NDArray[Any] | None = field(default=None)
    _react_softmax: npt.NDArray[Any] | None = field(default=None)

    @property
    def predicted(self) -> npt.NDArray[Any]:
        return np.argmax(self.softmax_output, axis=1)

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

    @property
    def features(self) -> npt.NDArray[Any] | None:
        return self._features

    @property
    def last_layer(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]] | None:
        return self._last_layer

    @property
    def vim_score(self):
        if self._vim_score is None:
            if self.features is None:
                return None

            self._vim_score = _vim(
                train_features=self._train_features,
                features=self.features,
                logits=self.logits,
                last_layer=self.last_layer,
            )

        return self._vim_score

    @property
    def maha_dist(self):
        if self._maha_dist is None:
            if self.features is None:
                return None

            self._maha_dist = _maha_dist(
                train_features=self._train_features,
                features=self.features,
                labels=self.labels,
                predicted=self.predicted,
                dataset_idx=self.dataset_idx,
            )

        return self._maha_dist

    @property
    def dknn_dist(self):
        if self._dknn_dist is None:
            if self.features is None:
                return None

            self._dknn_dist = _deep_knn(
                train_features=self._train_features,
                features=self.features,
                labels=self.labels,
                predicted=self.predicted,
                dataset_idx=self.dataset_idx,
            )

        return self._dknn_dist

    @property
    def react_logits(self):
        if self._react_logits is None:
            if self.features is None:
                return None

            self._react_logits = _react(
                last_layer=self.last_layer,
                features=self.features,
                train_features=self._train_features,
                dataset_idx=self.dataset_idx,
            )

        return self._react_logits

    @property
    def react_softmax(self):
        if self.react_logits is None:
            return None

        if self._react_softmax is None:
            self._react_softmax = scpspecial.softmax(
                self.react_logits.astype(np.float64), axis=1
            )

        return self._react_softmax

    def dataset_name_to_idx(self, dataset_name: str) -> int:
        if dataset_name == "val_tuning":
            return 0

        flat_test_set_list = []
        for _, datasets in self.config.eval.query_studies:
            if isinstance(datasets, (list, ListConfig)):
                if len(datasets) > 0:
                    if isinstance(datasets[0], configs.DataConfig):
                        datasets = map(lambda d: d.dataset, datasets)
                    flat_test_set_list.extend(list(datasets))
            elif (
                isinstance(datasets, configs.DataConfig)
                and datasets.dataset is not None
            ):
                flat_test_set_list.append(datasets.dataset)
            elif isinstance(datasets, str):
                flat_test_set_list.append(datasets)

        logger.error(f"{flat_test_set_list=}")

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
            _react_logits=_filter_if_exists(self.react_logits),
            _maha_dist=_filter_if_exists(self.maha_dist),
            _dknn_dist=_filter_if_exists(self.dknn_dist),
            _vim_score=_filter_if_exists(self.vim_score),
            mcd_external_confids_dist=_filter_if_exists(self.mcd_external_confids_dist),
            config=self.config,
            _train_features=self._train_features,
        )

    @staticmethod
    def __load_npz_if_exists(path: Path) -> npt.NDArray[np.float64] | None:
        if not path.is_file():
            return None

        with np.load(path) as npz:
            return npz.f.arr_0

    @overload
    @staticmethod
    def __load_from_store(
        config: configs.Config, file: str
    ) -> npt.NDArray[np.float64] | None:
        ...

    @overload
    @staticmethod
    def __load_from_store(
        config: configs.Config, file: str, dtype: type, unpack: Literal[False]
    ) -> dict[str, npt.NDArray[np.float64]] | None:
        ...

    @overload
    @staticmethod
    def __load_from_store(
        config: configs.Config, file: str, dtype: type
    ) -> npt.NDArray[np.float64] | None:
        ...

    @staticmethod
    def __load_from_store(
        config: configs.Config, file: str, dtype: type = np.float64, unpack: bool = True
    ) -> npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]] | None:
        store_paths = map(Path, os.getenv("FD_SHIFTS_STORE_PATH", "").split(":"))

        test_dir = config.test.dir.relative_to(os.getenv("EXPERIMENT_ROOT_DIR", ""))

        for store_path in store_paths:
            if (store_path / test_dir / file).is_file():
                logger.debug(f"Loading {store_path / test_dir / file}")
                with np.load(store_path / test_dir / file) as npz:
                    if unpack:
                        return npz.f.arr_0.astype(dtype)
                    else:
                        return dict(npz.items())

        return None

    @staticmethod
    def from_experiment(
        test_dir: Path,
        config: configs.Config,
        holdout_classes: list | None = None,
    ) -> ExperimentData:
        from fd_shifts.loaders.dataset_collection import CorruptCIFAR

        if not isinstance(test_dir, Path):
            test_dir = Path(test_dir)

        if (
            raw_output := ExperimentData.__load_from_store(config, "raw_logits.npz")
        ) is not None:
            logits = raw_output[:, :-2]
            softmax = scpspecial.softmax(logits, axis=1)

            if any(
                "mcd" in confid for confid in config.eval.confidence_measures.test
            ) and (
                (
                    mcd_logits_dist := ExperimentData.__load_from_store(
                        config, "raw_logits_dist.npz", dtype=np.float16
                    )
                )
                is not None
            ):
                if mcd_logits_dist.shape[0] > logits.shape[0]:
                    dset = CorruptCIFAR(
                        config.eval.query_studies.noise_study.data_dir,
                        train=False,
                        download=False,
                    )
                    idx = (
                        CorruptCIFAR.subsample_idx(
                            dset.data,
                            dset.targets,
                            config.eval.query_studies.noise_study.subsample_corruptions,
                        )
                        + raw_output[raw_output[:, -1] < 2].shape[0]
                    )
                    idx = np.concatenate(
                        [
                            np.argwhere(raw_output[:, -1] < 2).flatten(),
                            idx,
                            np.argwhere(raw_output[:, -1] > 2).flatten()
                            + mcd_logits_dist.shape[0]
                            - raw_output.shape[0],
                        ]
                    )
                    mcd_logits_dist = mcd_logits_dist[idx]
                mcd_logits_dist = mcd_logits_dist.astype(np.float64)
                mcd_softmax_dist = scpspecial.softmax(mcd_logits_dist, axis=1)
            else:
                mcd_logits_dist = None
                mcd_softmax_dist = None

        elif (
            raw_output := ExperimentData.__load_from_store(config, "raw_output.npz")
        ) is not None:
            logits = None
            mcd_logits_dist = None
            softmax = raw_output[:, :-2]
            mcd_softmax_dist = ExperimentData.__load_from_store(
                config, "raw_output_dist.npz"
            )
        else:
            raise FileNotFoundError(f"Could not find model output in {test_dir}")

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

        external_confids = ExperimentData.__load_from_store(
            config, "external_confids.npz"
        )
        if (
            any("mcd" in confid for confid in config.eval.confidence_measures.test)
            and (
                mcd_external_confids_dist := ExperimentData.__load_from_store(
                    config, "external_confids_dist.npz", dtype=np.float16
                )
            )
            is not None
        ):
            if mcd_external_confids_dist.shape[0] > logits.shape[0]:
                dset = CorruptCIFAR(
                    config.eval.query_studies.noise_study.data_dir,
                    train=False,
                    download=False,
                )
                idx = (
                    CorruptCIFAR.subsample_idx(
                        dset.data,
                        dset.targets,
                        config.eval.query_studies.noise_study.subsample_corruptions,
                    )
                    + raw_output[raw_output[:, -1] < 2].shape[0]
                )
                idx = np.concatenate(
                    [
                        np.argwhere(raw_output[:, -1] < 2).flatten(),
                        idx,
                        np.argwhere(raw_output[:, -1] > 2).flatten()
                        + mcd_logits_dist.shape[0]
                        - raw_output.shape[0],
                    ]
                )
                mcd_external_confids_dist = mcd_external_confids_dist[idx]
            mcd_external_confids_dist = mcd_external_confids_dist.astype(np.float64)
        else:
            mcd_external_confids_dist = None

        if (
            features := ExperimentData.__load_from_store(config, "encoded_output.npz")
        ) is not None:
            features = features[:, :-1]

        if (
            last_layer := ExperimentData.__load_from_store(
                config, "last_layer.npz", unpack=False
            )
        ) is not None:
            last_layer = tuple(last_layer.values())

        train_features = ExperimentData.__load_from_store(config, "train_features.npz")

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
            _features=features,
            _train_features=train_features,
            _last_layer=last_layer,  # type: ignore
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


class TemperatureScaling:
    def __init__(self, val_logits: npt.NDArray[Any], val_labels: npt.NDArray[Any]):
        import torch

        logger.info("Fit temperature to validation logits")
        self.temperature = torch.ones(1).requires_grad_(True)

        logits = torch.tensor(val_logits)
        labels = torch.tensor(val_labels).long()

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def _eval():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)  # type: ignore

        self.temperature = self.temperature.item()

    def __call__(self, logits: npt.NDArray[Any]) -> npt.NDArray[Any]:
        import torch

        return np.max(
            torch.softmax(torch.tensor(logits) / self.temperature, dim=1).numpy(),
            axis=1,
        )


def _react(
    last_layer: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
    train_features: npt.NDArray[np.float_] | None,
    features: npt.NDArray[np.float_],
    dataset_idx: npt.NDArray[np.integer],
    clip_quantile=99,
    val_set_index=0,
    is_dg=False,
):
    import torch

    logger.info("Compute REACT logits")

    clip = torch.tensor(np.quantile(train_features[:, :-1], clip_quantile / 100))

    w, b = last_layer
    w = torch.tensor(w, dtype=torch.float)
    w = torch.tensor(w, dtype=torch.float)

    logits = (
        torch.matmul(
            torch.clip(torch.tensor(features, dtype=torch.float), min=None, max=clip),
            w.T,
        )
        + b
    )
    if is_dg:
        logits = logits[:, :-1]
    return logits.numpy()


def _maha_dist(
    train_features: npt.NDArray[np.float_] | None,
    features: npt.NDArray[np.float_],
    labels: npt.NDArray[np.int_],
    predicted: npt.NDArray[np.int_],
    dataset_idx: npt.NDArray[np.int_],
):
    import torch

    logger.info("Compute Mahalanobis distance")

    val_features = train_features[:, :-1]
    val_labels = train_features[:, -1]

    means = torch.tensor(
        np.array(
            [val_features[val_labels == i].mean(axis=0) for i in np.unique(val_labels)]
        )
    )
    icov = torch.pinverse(torch.cov(torch.tensor(val_features).float().T))

    tpredicted = torch.tensor(predicted).long()
    zm = torch.tensor(features) - means[tpredicted]
    zm = zm.float()

    maha = -(torch.einsum("ij,jk,ik->i", zm, icov, zm))
    maha = maha.numpy()
    return maha


def _vim(
    last_layer: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
    train_features: npt.NDArray[np.float_] | None,
    features: npt.NDArray[np.float_],
    logits: npt.NDArray[np.float_],
    is_dg=False,
):
    import torch

    logger.info("Compute ViM score")
    if features.shape[-1] >= 2048:
        D = 1000
    elif features.shape[-1] >= 768:
        D = 512
    else:
        D = features.shape[-1] // 2

    w, b = last_layer
    w = torch.tensor(w, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)

    logger.debug("ViM: Compute NS")
    u = -torch.pinverse(w) @ b
    train_f = torch.tensor(train_features[:, :-1], dtype=torch.float)
    cov = torch.cov((train_f - u).T)
    eig_vals, eigen_vectors = torch.linalg.eig(cov)
    eig_vals = eig_vals.real
    eigen_vectors = eigen_vectors.real
    NS = (eigen_vectors.T[torch.argsort(eig_vals * -1)[D:]]).T

    logger.debug("ViM: Compute alpha")
    logit_train = torch.matmul(train_f, w.T) + b

    if is_dg:
        logit_train = logit_train[:, :-1]

    vlogit_train = torch.linalg.norm(torch.matmul(train_f - u, NS), dim=-1)
    alpha = logit_train.max(dim=-1)[0].mean() / vlogit_train.mean()

    tlogits = torch.tensor(logits, dtype=torch.float)
    tfeatures = torch.tensor(features, dtype=torch.float)

    logger.debug("ViM: Compute score")
    energy = torch.logsumexp(tlogits, dim=-1)
    vlogit = torch.linalg.norm(torch.matmul(tfeatures - u, NS), dim=-1) * alpha
    score = -vlogit + energy
    return score.numpy()


def _deep_knn(
    train_features: npt.NDArray[np.float_] | None,
    features: npt.NDArray[np.float_],
    labels: npt.NDArray[np.int_],
    predicted: npt.NDArray[np.int_],
    dataset_idx: npt.NDArray[np.int_],
    val_set_index=0,
):
    logger.info("Compute DeepKNN distance")
    # index = faiss.IndexFlatL2(ftrain.shape[1])
    # index.add(ftrain)
    K = 50
    # neigh = neighbors.NearestNeighbors(n_neighbors=K, metric="euclidean", n_jobs=-1)
    # neigh.fit(train_features[:, :-1])
    # D, _ = neigh.kneighbors(features, return_distance=True)

    train_features = train_features[:1000, :-1]
    index = faiss.IndexFlatL2(train_features.shape[1])
    index.add(train_features.astype(np.float32))
    D, _ = index.search(features.astype(np.float32), K)

    score = -D[:, -1]
    return score


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
                    lambda confid: "mcd" not in confid and "waic" not in confid,
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
            self.method_dict["query_confids"].append("temp_mls")
            self.method_dict["query_confids"].append("energy_mls")

        if (
            self.experiment_data.features is not None
            and self.experiment_data.last_layer is not None
        ):
            self.method_dict["query_confids"].append("maha")
            self.method_dict["query_confids"].append("dknn")
            self.method_dict["query_confids"].append("vim")
            self.method_dict["query_confids"].append("react_det_mcp")
            self.method_dict["query_confids"].append("react_det_mls")
            self.method_dict["query_confids"].append("react_temp_mls")
            self.method_dict["query_confids"].append("react_energy_mls")

        if self.experiment_data.mcd_logits_dist is not None:
            self.method_dict["query_confids"].append("mcd_mls")

        logger.debug("CSFs: {}", ", ".join(self.method_dict["query_confids"]))

        self.query_performance_metrics = query_performance_metrics
        self.query_confid_metrics = query_confid_metrics
        self.query_plots = query_plots
        self.query_studies = (
            self.cfg.eval.query_studies if query_studies is None else query_studies
        )
        for study_name, datasets in self.query_studies:
            if isinstance(datasets, (list, ListConfig)) and len(datasets) > 0:
                if isinstance(datasets[0], configs.DataConfig):
                    self.query_studies.__dict__[study_name] = list(
                        map(
                            lambda d: d.dataset
                            + (
                                "_384"
                                if d.img_size[0] == 384 and "384" not in d.dataset
                                else ""
                            ),
                            datasets,
                        )
                    )
            if isinstance(datasets, configs.DataConfig):
                if datasets.dataset is not None:
                    self.query_studies.__dict__[study_name] = [
                        datasets.dataset
                        + (
                            "_384"
                            if datasets.img_size[0] == 384
                            and "384" not in datasets.dataset
                            else ""
                        )
                    ]
                else:
                    self.query_studies.__dict__[study_name] = []

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
        logger.info("Performing study {}", study_name)
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
            logger.debug(f"Compute score {query_confid}")
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
                elif "temp_mls" in query_confid:
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
            logger.debug("{}: evaluating {}", self.study_name, confid_key)
            confid_dict = self.method_dict[confid_key]

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
                    plot_val_risk_scores = eval.get_val_risk_scores(
                        self.rstar, self.rdelta
                    )
                    self.plot_threshs.append(plot_val_risk_scores["theta"])
                    self.true_covs.append(plot_val_risk_scores["val_cov"])

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
                            f"Noise sanity check: {corr_ix=}, {self.dummy_noise_ixs[ix]=}"
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
        logger.info(
            "Saved csv to {}",
            os.path.join(
                self.analysis_out_dir, "analysis_metrics_{}.csv".format(self.study_name)
            ),
        )

        # group_file_path = os.path.join(
        #     self.cfg.exp.group_dir, "group_analysis_metrics.csv"
        # )
        # if os.path.exists(group_file_path):
        #     with open(group_file_path, "a") as f:
        #         df.to_csv(f, float_format="%.5f", decimal=".", header=False)
        # else:
        #     with open(group_file_path, "w") as f:
        #         df.to_csv(f, float_format="%.5f", decimal=".")

    def _create_threshold_plot(self):
        f = ThresholdPlot(self.threshold_plot_dict)
        f.savefig(
            os.path.join(
                self.analysis_out_dir,
                "threshold_plot_{}.png".format(self.threshold_plot_confid),
            )
        )
        logger.info(
            "Saved threshold_plot to {}",
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
        logger.info(
            "Saved masterplot to {}",
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

    logger.info(
        "Starting analysis with in_path {}, out_path {}, and query studies {}".format(
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
