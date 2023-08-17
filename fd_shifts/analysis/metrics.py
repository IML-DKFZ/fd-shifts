from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt
from sklearn import calibration as skc
from sklearn import metrics as skm
from sklearn import preprocessing as skp
from sklearn import utils as sku
from typing_extensions import ParamSpec

from . import logger

AURC_DISPLAY_SCALE = 1000

_metric_funcs = {}

T = TypeVar("T")
P = ParamSpec("P")


def may_raise_sklearn_exception(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to handle sklearn exceptions

    Args:
        func (Callable[P, T]): function that may raise sklearn exception

    Returns:
        wrapped function
    """

    @logger.catch(ValueError, level=logging.DEBUG, default=np.nan)
    def _inner_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)

    return _inner_wrapper


@dataclass
class StatsCache:
    """Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values
        correct (array_like): Boolean array (best converted to int) where predictions were correct
    """

    confids: npt.NDArray[Any]
    correct: npt.NDArray[Any]
    n_bins: int
    labels: npt.NDArray[Any] | None = None

    @cached_property
    def roc_curve_stats(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        fpr, tpr, _ = skm.roc_curve(self.correct, self.confids)
        return fpr, tpr

    @property
    def residuals(self) -> npt.NDArray[Any]:
        return 1 - self.correct

    @cached_property
    def brc_curve_stats(self) -> tuple[list[float], list[float], list[float]]:
        coverages = []
        balanced_risks = []
        error_per_class = {}
        risk_per_class = {}
        assert self.labels is not None, "labels must be set"
        assert len(self.labels) == len(
            self.confids
        ), "labels must be same size as confids"
        # calculate risk per class
        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)
        n_remaining_per_class = {}
        # coverage = number samples
        coverage = n_residuals
        # calcualte baselines:
        # errors per class, residuals per class (total amount of images/errors from that class)
        # risk per class: errors per class by remaining images in this class
        # if there are no more images of a class risk is set to None and then filtered out before calculating mean
        for label in np.unique(self.labels):
            # remaining_labels = self.labels[idx_sorted]
            idx_class = np.where(self.labels == label)[0]
            error_per_class[label] = sum(self.residuals[idx_class])
            n_remaining_per_class[label] = len(idx_class)
            if n_remaining_per_class[label] == 0:
                risk_per_class[label] = None
            else:
                risk_per_class[label] = (
                    error_per_class[label] / n_remaining_per_class[label]
                )
        # coverage and risk point on the curve. starting point
        coverages.append(coverage / n_residuals)
        balanced_risks.append(
            np.array([x for x in risk_per_class.values() if x is not None]).mean()
        )
        weights = []
        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            # Decide which class the images is taken from
            label = int(self.labels[idx_sorted[i]])
            # from that class subtract 1 if an error is taken out and 0 if no error is taken out
            error_per_class[label] = (
                error_per_class[label] - self.residuals[idx_sorted[i]]
            )
            # reduce the remaining amount of images in the class an images was taken out
            n_remaining_per_class[label] = n_remaining_per_class[label] - 1
            assert (
                n_remaining_per_class[label] >= 0
            ), "Remaining images should be larger 0"
            # if there is one or no more images remaining in a class risk is set to 0
            # otherwise risk of the class is errors remaining divided by number images remaining
            if n_remaining_per_class[label] < 1:
                risk_per_class[label] = None
            else:
                risk_per_class[label] = error_per_class[label] / (
                    n_remaining_per_class[label]
                )
                assert risk_per_class[label] >= 0, "Risk can never be below 0"
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                balanced_risks.append(
                    np.array(
                        [x for x in risk_per_class.values() if x is not None]
                    ).mean()
                )
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0
        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            balanced_risks.append(balanced_risks[-1])
            weights.append(tmp_weight / n_residuals)
        return coverages, balanced_risks, weights

    @cached_property
    def rc_curve_stats(self) -> tuple[list[float], list[float], list[float]]:
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)

        return coverages, risks, weights

    @cached_property
    def calibration_stats(self):
        """Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/calibration.py#L869
        """
        calib_confids = np.clip(self.confids, 0, 1)

        n_bins = self.n_bins
        y_true = sku.column_or_1d(self.correct)
        y_prob = sku.column_or_1d(calib_confids)

        if y_prob.min() < 0 or y_prob.max() > 1:
            raise ValueError(
                "y_prob has values outside [0, 1] and normalize is " "set to False."
            )

        labels = np.unique(y_true)
        if len(labels) > 2:
            raise ValueError(
                "Only binary classification is supported. " f"Provided labels {labels}."
            )
        y_true = skp.label_binarize(y_true, classes=labels)[:, 0]

        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

        binids = np.digitize(y_prob, bins) - 1

        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        prob_total = bin_total[nonzero] / bin_total.sum()

        return prob_total, prob_true, prob_pred

    @cached_property
    def bin_discrepancies(self):
        _, bin_accs, bin_confids = self.calibration_stats
        return np.abs(bin_accs - bin_confids)


def register_metric_func(name: str) -> Callable:
    """Decorator to register a new metric function

    Args:
        name (str): name to register under

    Returns:
        registered callable
    """

    def _inner_wrapper(func: Callable) -> Callable:
        _metric_funcs[name] = func
        return func

    return _inner_wrapper


def metric_function_exists(metric_name: str) -> bool:
    """Check if metric function exists

    Args:
        metric_name (str): name of the metric function

    Returns:
        True if it exists, False otherwise
    """
    return metric_name in _metric_funcs


def get_metric_function(metric_name: str) -> Callable[[StatsCache], float]:
    """Get the metric function callable registered under metric_name

    Args:
        metric_name (str): name of the metric function

    Returns:
        callable
    """
    if not metric_function_exists(metric_name):
        return _metric_funcs["*"]

    return _metric_funcs[metric_name]


@register_metric_func("failauc")
@may_raise_sklearn_exception
def failauc(stats_cache: StatsCache) -> float:
    """AUROC_f metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    fpr, tpr = stats_cache.roc_curve_stats
    return skm.auc(fpr, tpr)


@register_metric_func("fpr@95tpr")
@may_raise_sklearn_exception
def fpr_at_95_tpr(stats_cache: StatsCache) -> float:
    """False Positive Rate at 95% true positive rate metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    fpr, tpr = stats_cache.roc_curve_stats
    return np.min(fpr[np.argwhere(tpr >= 0.9495)])


@register_metric_func("failap_suc")
@may_raise_sklearn_exception
def failap_suc(stats_cache: StatsCache) -> float:
    """Average Precision for success metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    return cast(
        float,
        skm.average_precision_score(
            stats_cache.correct,
            stats_cache.confids,
            pos_label=1,
        ),
    )


@register_metric_func("failap_err")
@may_raise_sklearn_exception
def failap_err(stats_cache: StatsCache) -> float:
    """Average Precision for error metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    return cast(
        float,
        skm.average_precision_score(
            stats_cache.correct, -stats_cache.confids, pos_label=0
        ),
    )


@register_metric_func("aurc")
@may_raise_sklearn_exception
def aurc(stats_cache: StatsCache) -> float:
    """AURC metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    _, risks, weights = stats_cache.rc_curve_stats
    return (
        sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])
        * AURC_DISPLAY_SCALE
    )


@register_metric_func("b-aurc")
@may_raise_sklearn_exception
def baurc(stats_cache: StatsCache):
    _, risks, weights = stats_cache.brc_curve_stats
    return (
        sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])
        * AURC_DISPLAY_SCALE
    )


@register_metric_func("e-aurc")
@may_raise_sklearn_exception
def eaurc(stats_cache: StatsCache) -> float:
    """eAURC metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    err = np.mean(stats_cache.residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err + np.finfo(err.dtype).eps))
    return aurc(stats_cache) - kappa_star_aurc * AURC_DISPLAY_SCALE


@register_metric_func("mce")
@may_raise_sklearn_exception
def maximum_calibration_error(stats_cache: StatsCache) -> float:
    """Maximum Calibration Error metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    return (stats_cache.bin_discrepancies).max()


@register_metric_func("ece")
@may_raise_sklearn_exception
def expected_calibration_error(stats_cache: StatsCache) -> float:
    """Expected Calibration Error metric function

    See reference
    https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/stats/calibration.py#L258-L319

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    prob_total, _, _ = stats_cache.calibration_stats
    return np.dot(stats_cache.bin_discrepancies, prob_total)


@register_metric_func("fail-NLL")
@may_raise_sklearn_exception
def failnll(stats_cache: StatsCache) -> float:
    """NLL metric function

    Args:
        stats_cache (StatsCache): StatsCache object

    Returns:
        metric value
    """
    return -np.mean(
        stats_cache.correct * np.log(stats_cache.confids + 1e-7)
        + (1 - stats_cache.correct) * np.log(1 - stats_cache.confids + 1e-7)
    )
