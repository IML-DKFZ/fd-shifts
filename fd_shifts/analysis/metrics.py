from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt
from sklearn import calibration as skc
from sklearn import metrics as skm
from typing_extensions import ParamSpec

AURC_DISPLAY_SCALE = 1000

_metric_funcs = {}

T = TypeVar("T")
P = ParamSpec("P")


def may_raise_sklearn_exception(func: Callable[P, T]) -> Callable[P, T]:
    def _inner_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except ValueError:
            return cast(T, np.nan)

    return _inner_wrapper


@dataclass
class StatsCache:
    """ Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values
        correct (array_like): Boolean array (best converted to int) where predictions were correct
    """

    confids: npt.NDArray[Any]
    correct: npt.NDArray[Any]
    n_bins: int

    @cached_property
    def roc_curve_stats(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        fpr, tpr, _ = skm.roc_curve(self.correct, self.confids)
        return fpr, tpr

    @property
    def residuals(self) -> npt.NDArray[Any]:
        return 1 - self.correct

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
        calib_confids = np.clip(self.confids, 0, 1)  # necessary for waic
        bin_accs, bin_confids = skc.calibration_curve(
            self.correct, calib_confids, n_bins=self.n_bins
        )

        return bin_accs, bin_confids

    @cached_property
    def hist_confids(self):
        return np.histogram(self.confids, bins=self.n_bins, range=(0, 1))[0]

    @cached_property
    def bin_discrepancies(self):
        bin_accs, bin_confids = self.calibration_stats
        return np.abs(bin_accs - bin_confids)


def register_metric_func(name: str) -> Callable:
    def _inner_wrapper(func: Callable) -> Callable:
        _metric_funcs[name] = func
        return func

    return _inner_wrapper


def get_metric_function(metric_name: str) -> Callable[[StatsCache], float]:
    if metric_name not in _metric_funcs:
        return _metric_funcs["*"]

    return _metric_funcs[metric_name]


@register_metric_func("failauc")
@may_raise_sklearn_exception
def failauc(stats_cache: StatsCache) -> float:
    fpr, tpr = stats_cache.roc_curve_stats
    return skm.auc(fpr, tpr)


@register_metric_func("fpr@95tpr")
@may_raise_sklearn_exception
def fpr_at_95_tpr(stats_cache: StatsCache) -> float:
    fpr, tpr = stats_cache.roc_curve_stats
    return np.min(fpr[np.argwhere(tpr >= 0.9495)])


@register_metric_func("failap_suc")
@may_raise_sklearn_exception
def failap_suc(stats_cache: StatsCache) -> float:
    return cast(
        float,
        skm.average_precision_score(
            stats_cache.correct, stats_cache.confids, pos_label=1
        ),
    )


@register_metric_func("failap_err")
@may_raise_sklearn_exception
def failap_err(stats_cache: StatsCache):
    return cast(
        float,
        skm.average_precision_score(
            stats_cache.correct, -stats_cache.confids, pos_label=0
        ),
    )


@register_metric_func("aurc")
@may_raise_sklearn_exception
def aurc(stats_cache: StatsCache):
    _, risks, weights = stats_cache.rc_curve_stats
    return (
        sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])
        * AURC_DISPLAY_SCALE
    )


@register_metric_func("e-aurc")
@may_raise_sklearn_exception
def eaurc(stats_cache: StatsCache):
    err = np.mean(stats_cache.residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    return aurc(stats_cache) - kappa_star_aurc * AURC_DISPLAY_SCALE


@register_metric_func("mce")
@may_raise_sklearn_exception
def maximum_calibration_error(stats_cache: StatsCache):
    return (stats_cache.bin_discrepancies).max()


@register_metric_func("ece")
@may_raise_sklearn_exception
def expected_calibration_error(stats_cache: StatsCache):
    # BUG: Check length of bin_discrepancies and non-zero hist_confids
    return (
        np.dot(
            stats_cache.bin_discrepancies,
            stats_cache.hist_confids[np.argwhere(stats_cache.hist_confids > 0)],
        )
        / np.sum(stats_cache.hist_confids)
    )[0]


@register_metric_func("fail-NLL")
@may_raise_sklearn_exception
def failnll(stats_cache: StatsCache):
    return -np.mean(
        stats_cache.correct * np.log(stats_cache.confids + 1e-7)
        + (1 - stats_cache.correct) * np.log(1 - stats_cache.confids + 1e-7)
    )
