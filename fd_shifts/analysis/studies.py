from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Iterator, Tuple

import numpy as np
import numpy.typing as npt

from . import logger

if TYPE_CHECKING:
    from fd_shifts.analysis import Analysis, ExperimentData

_filter_funcs = {}


def register_filter_func(name: str) -> Callable:
    """Decorator to register a new filter function

    Args:
        name (str): name to register under

    Returns:
        registered callable
    """

    def _inner_wrapper(func: Callable) -> Callable:
        _filter_funcs[name] = func
        return func

    return _inner_wrapper


def get_filter_function(study_name: str) -> Callable:
    """Get the filter callable registered under study_name

    Args:
        study_name (str): name of the study

    Returns:
        callable
    """
    if study_name not in _filter_funcs:
        return _filter_funcs["*"]

    return _filter_funcs[study_name]


_study_iterators = {}


def register_study_iterator(name: str) -> Callable:
    """Decorator to register a new study iterator

    Args:
        name (str): name to register under

    Returns:
        registered callable
    """

    def _inner_wrapper(func: Callable) -> Callable:
        _study_iterators[name] = func
        return func

    return _inner_wrapper


def get_study_iterator(study_name: str) -> Callable:
    """Get the study iterator registered under confid_name

    Args:
        study_name (str): name of the study

    Returns:
        callable
    """
    if study_name not in _study_iterators:
        return _study_iterators["*"]

    return _study_iterators[study_name]


@register_filter_func("*")
def filter_data(data: "ExperimentData", dataset_name: str) -> "ExperimentData":
    """Filter experiment data by dataset

    Args:
        data (ExperimentData): unfiltered experiment data
        dataset_name (str): dataset to filter by

    Returns:
        filtered experiment data
    """
    return data.filter_dataset_by_name(dataset_name)


@register_study_iterator("*")
def iterate_default_study(
    study_name: str, analysis: "Analysis"
) -> Iterator[Tuple[str, "ExperimentData"]]:
    """Generic iterator over filtered experiment data based on studies to run

    Args:
        study_name (str): name of the study
        analysis (Analysis): analysis wrapper object

    Yields:
        tuples of study names and filtered data
    """
    filter_func: Callable[..., "ExperimentData"] = get_filter_function(study_name)
    study_data = filter_func(
        analysis.experiment_data,
        "val_tuning"
        if study_name == "val_tuning"
        else getattr(analysis.query_studies, study_name),
    )

    yield study_name, study_data


@register_study_iterator("in_class_study")
def iterate_in_class_study_data(
    study_name: str, analysis: "Analysis"
) -> Iterator[Tuple[str, "ExperimentData"]]:
    """Iterator over filtered experiment data based on sub-class shift datasets

    Args:
        study_name (str): name of the study
        analysis (Analysis): analysis wrapper object

    Yields:
        tuples of study names and filtered data
    """
    filter_func: Callable[..., "ExperimentData"] = get_filter_function(study_name)
    for in_class_set in getattr(analysis.query_studies, study_name):
        study_data = filter_func(
            analysis.experiment_data,
            in_class_set,
        )

        yield f"{study_name}_{in_class_set}", study_data


@register_filter_func("new_class_study")
def filter_new_class_study_data(
    data: "ExperimentData",
    iid_set_name: str,
    dataset_name: str,
    mode: str = "proposed_mode",
) -> "ExperimentData":
    """Filter experiment data by new-class dataset and set correct array

    Args:
        data (ExperimentData): unfiltered experiment data
        iid_set_name (str): iid dataset
        dataset_name (str): dataset to filter by
        mode (str): handling of iid missclassifications

    Returns:
        filtered experiment data
    """
    iid_set_ix = data.dataset_name_to_idx(iid_set_name)
    new_class_set_ix = data.dataset_name_to_idx(dataset_name)

    select_ix_in = np.argwhere(data.dataset_idx == iid_set_ix)[:, 0]
    select_ix_out = np.argwhere(data.dataset_idx == new_class_set_ix)[:, 0]

    assert data.correct is not None

    correct = deepcopy(data.correct)
    correct[select_ix_out] = 0
    if mode == "original_mode":
        correct[
            select_ix_in
        ] = 1  # nice to see so visual how little practical sense the current protocol makes!
    labels = deepcopy(data.labels)
    labels[select_ix_out] = -99

    select_ix_all = np.argwhere(
        (data.dataset_idx == new_class_set_ix)
        | ((data.dataset_idx == iid_set_ix) & (correct == 1))
    )[
        :, 0
    ]  # de-select incorrect inlier predictions.

    mcd_correct = deepcopy(data.mcd_correct)
    if mcd_correct is not None:
        mcd_correct[select_ix_out] = 0
        if mode == "original_mode":
            mcd_correct[select_ix_in] = 1

        select_ix_all_mcd = np.argwhere(
            (data.dataset_idx == new_class_set_ix)
            | ((data.dataset_idx == iid_set_ix) & (mcd_correct == 1))
        )[:, 0]
    else:
        select_ix_all_mcd = None

    def __filter_if_exists(data: npt.NDArray[Any] | None, mask):
        if data is not None:
            return data[mask]

        return None

    return data.__class__(
        softmax_output=data.softmax_output[select_ix_all],
        logits=__filter_if_exists(data.logits, select_ix_all),
        labels=labels[select_ix_all],
        dataset_idx=data.dataset_idx[select_ix_all],
        mcd_softmax_dist=__filter_if_exists(data.mcd_softmax_dist, select_ix_all_mcd),
        mcd_logits_dist=__filter_if_exists(data.mcd_logits_dist, select_ix_all_mcd),
        external_confids=__filter_if_exists(data.external_confids, select_ix_all),
        mcd_external_confids_dist=__filter_if_exists(
            data.mcd_external_confids_dist, select_ix_all_mcd
        ),
        config=data.config,
        _correct=__filter_if_exists(correct, select_ix_all),
        _mcd_correct=__filter_if_exists(mcd_correct, select_ix_all_mcd),
        _mcd_labels=__filter_if_exists(labels, select_ix_all_mcd),
    )


@register_study_iterator("new_class_study")
def iterate_new_class_study_data(
    study_name: str, analysis: "Analysis"
) -> Iterator[Tuple[str, "ExperimentData"]]:
    """Iterator over filtered experiment data based on new-class shift datasets

    Args:
        study_name (str): name of the study
        analysis (Analysis): analysis wrapper object

    Yields:
        tuples of study names and filtered data
    """
    filter_func: Callable[..., "ExperimentData"] = get_filter_function(study_name)
    for new_class_set in getattr(analysis.query_studies, study_name):
        for mode in ["original_mode", "proposed_mode"]:
            study_data = filter_func(
                analysis.experiment_data,
                analysis.query_studies.iid_study,
                new_class_set,
                mode,
            )

            yield f"{study_name}_{new_class_set}_{mode}", study_data


@register_filter_func("openset")
def filter_openset_study_data(
    data: "ExperimentData",
    iid_set_name: str,
    holdout_classes: list[int],
    mode: str = "proposed_mode",
) -> "ExperimentData":
    """Filter experiment data by openset dataset and set correct array

    Args:
        data (ExperimentData): unfiltered experiment data
        iid_set_name (str): iid dataset
        holdout_classes (list[int]): classes that were not trained on
        mode (str): handling of iid missclassifications

    Returns:
        filtered experiment data
    """
    select_ix_in = np.argwhere(np.isin(data.labels, holdout_classes, invert=True))[:, 0]
    select_ix_out = np.argwhere(np.isin(data.labels, holdout_classes))[:, 0]

    assert data.correct is not None

    correct = deepcopy(data.correct)
    correct[select_ix_out] = 0
    if mode == "original_mode":
        correct[
            select_ix_in
        ] = 1  # nice to see so visual how little practical sense the current protocol makes!
    labels = deepcopy(data.labels)

    select_ix_all = np.argwhere(
        np.isin(data.labels, holdout_classes)
        | (np.isin(data.labels, holdout_classes, invert=True) & (correct == 1))
    )[
        :, 0
    ]  # de-select incorrect inlier predictions.

    mcd_correct = deepcopy(data.mcd_correct)
    if mcd_correct is not None:
        mcd_correct[select_ix_out] = 0
        if mode == "original_mode":
            mcd_correct[select_ix_in] = 1

        select_ix_all_mcd = np.argwhere(
            np.isin(data.labels, holdout_classes)
            | (np.isin(data.labels, holdout_classes, invert=True) & (mcd_correct == 1))
        )[:, 0]
    else:
        select_ix_all_mcd = None

    def __filter_if_exists(data: npt.NDArray[Any] | None, mask):
        if data is not None:
            return data[mask]

        return None

    return data.__class__(
        softmax_output=data.softmax_output[select_ix_all],
        logits=__filter_if_exists(data.logits, select_ix_all),
        labels=labels[select_ix_all],
        dataset_idx=data.dataset_idx[select_ix_all],
        mcd_softmax_dist=__filter_if_exists(data.mcd_softmax_dist, select_ix_all_mcd),
        mcd_logits_dist=__filter_if_exists(data.mcd_logits_dist, select_ix_all_mcd),
        external_confids=__filter_if_exists(data.external_confids, select_ix_all),
        mcd_external_confids_dist=__filter_if_exists(
            data.mcd_external_confids_dist, select_ix_all_mcd
        ),
        config=data.config,
        _correct=__filter_if_exists(correct, select_ix_all),
        _mcd_correct=__filter_if_exists(mcd_correct, select_ix_all_mcd),
    )


@register_study_iterator("openset")
def iterate_openset_study_data(
    study_name: str, analysis: "Analysis"
) -> Iterator[Tuple[str, "ExperimentData"]]:
    """Iterator over filtered experiment data based on openset datasets

    Args:
        study_name (str): name of the study
        analysis (Analysis): analysis wrapper object

    Yields:
        tuples of study names and filtered data
    """
    filter_func: Callable[..., "ExperimentData"] = get_filter_function(study_name)
    for mode in ["original_mode", "proposed_mode"]:
        study_data = filter_func(
            analysis.experiment_data,
            analysis.query_studies.iid_study,
            analysis.holdout_classes,
            mode,
        )

        yield f"{study_name}_{mode}", study_data


@register_filter_func("noise_study")
def filter_noise_study_data(
    data: "ExperimentData",
    dataset_name: str,
    noise_level: int = 1,
    fast_dev_run: bool = False,
) -> "ExperimentData":
    """Filter experiment data by corruption shift dataset

    Args:
        data (ExperimentData): unfiltered experiment data
        dataset_name (str): dataset to filter by
        noise_level (int): level of corruption

    Returns:
        filtered experiment data
    """
    noise_set_ix = data.dataset_name_to_idx(dataset_name)

    select_ix = np.argwhere(data.dataset_idx == noise_set_ix)[:, 0]

    def __filter_intensity_3d(data, mask, noise_level):
        if data is None:
            return None

        data = data[mask]

        if fast_dev_run:
            return data

        return data.reshape(15, 5, -1, data.shape[-2], data.shape[-1])[
            :, noise_level
        ].reshape(-1, data.shape[-2], data.shape[-1])

    def __filter_intensity_2d(data, mask, noise_level):
        if data is None:
            return None

        data = data[mask]

        if fast_dev_run:
            return data

        return data.reshape(15, 5, -1, data.shape[-1])[:, noise_level].reshape(
            -1, data.shape[-1]
        )

    def __filter_intensity_1d(data, mask, noise_level):
        if data is None:
            return None

        data = data[mask]

        if fast_dev_run:
            return data

        return data.reshape(15, 5, -1)[:, noise_level].reshape(-1)

    return data.__class__(
        softmax_output=__filter_intensity_2d(
            data.softmax_output, select_ix, noise_level
        ),
        logits=__filter_intensity_2d(data.logits, select_ix, noise_level),
        labels=__filter_intensity_1d(data.labels, select_ix, noise_level),
        dataset_idx=__filter_intensity_1d(data.dataset_idx, select_ix, noise_level),
        external_confids=__filter_intensity_1d(
            data.external_confids, select_ix, noise_level
        ),
        mcd_external_confids_dist=__filter_intensity_2d(
            data.mcd_external_confids_dist, select_ix, noise_level
        ),
        mcd_softmax_dist=__filter_intensity_3d(
            data.mcd_softmax_dist, select_ix, noise_level
        ),
        mcd_logits_dist=__filter_intensity_3d(
            data.mcd_logits_dist, select_ix, noise_level
        ),
        config=data.config,
    )


@register_study_iterator("noise_study")
def iterate_noise_study_data(
    study_name: str, analysis: "Analysis"
) -> Iterator[Tuple[str, "ExperimentData"]]:
    """Iterator over filtered experiment data based on corruption shift datasets

    Args:
        study_name (str): name of the study
        analysis (Analysis): analysis wrapper object

    Yields:
        tuples of study names and filtered data
    """
    filter_func: Callable[..., "ExperimentData"] = get_filter_function(study_name)
    for noise_set in getattr(analysis.query_studies, study_name):
        for intensity_level in range(5):
            logger.debug(
                "starting noise study with intensitiy level %s",
                intensity_level + 1,
            )

            study_data = filter_func(
                analysis.experiment_data,
                noise_set,
                intensity_level,
                analysis.cfg.trainer.fast_dev_run,
            )

            yield f"{study_name}_{intensity_level + 1}", study_data
