from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from fd_shifts.analysis import Analysis, ExperimentData

EXTERNAL_CONFIDS = ["ext", "bpd", "maha", "tcp", "dg", "devries"]

ArrayType = npt.NDArray[np.floating]
T = TypeVar(
    "T", Callable[[ArrayType], ArrayType], Callable[[ArrayType, ArrayType], ArrayType]
)


def _assert_softmax_finite(softmax: npt.NDArray[np.number]):
    assert np.isfinite(softmax).all(), "NaN or INF in softmax output"


def _assert_softmax_numerically_stable(softmax: ArrayType):
    msr = softmax.max(axis=1)
    errors = (msr == 1) & ((softmax > 0) & (softmax < 1)).any(axis=1)

    if softmax.dtype != np.float64:
        logging.warning("Softmax is not 64bit, not checking for numerical stability")
        return

    # alert if more than 10% are erroneous
    assert (
        errors.mean() < 0.1
    ), f"Numerical errors in softmax: {errors.mean() * 100:.2f}%"


def validate_softmax(func: T) -> T:
    """Decorator to validate softmax before computing stuff

    Args:
        func (T): callable to decorate

    Returns:
        decorated callable
    """

    def _inner_wrapper(*args: ArrayType) -> ArrayType:
        for arg in args:
            _assert_softmax_finite(arg)
            _assert_softmax_numerically_stable(arg)
        return func(*args)

    return cast(T, _inner_wrapper)


def is_external_confid(name: str):
    return name.split("_")[0] in EXTERNAL_CONFIDS


def is_mcd_confid(name: str):
    return "mcd" in name or "waic" in name


_confid_funcs = {}


def register_confid_func(name: str) -> Callable:
    """Decorator to register a new confidence scoring function

    Args:
        name (str): name to register under

    Returns:
        registered callable
    """

    def _inner_wrapper(func: Callable) -> Callable:
        _confid_funcs[name] = func
        return func

    return _inner_wrapper


def confid_function_exists(confid_name: str) -> bool:
    """Check if confidence scoring function exists

    Args:
        confid_name (str): name of the CSF

    Returns:
        True if it exists, False otherwise
    """
    return confid_name in _confid_funcs


def get_confid_function(confid_name) -> Callable:
    """Get the CSF callable registered under confid_name

    Args:
        confid_name (str): name of the CSF

    Returns:
        callable
    """
    if not confid_function_exists(confid_name):
        raise NotImplementedError(f"Function for {confid_name} not implemented.")

    return _confid_funcs[confid_name]


@register_confid_func("det_mcp")
@validate_softmax
@register_confid_func("det_mls")
def maximum_softmax_probability(
    softmax: ArrayType,
) -> ArrayType:
    """Maximum softmax probability CSF/Maximum logit score CSF

    Args:
        softmax (ArrayType): array-like containing softmaxes or logits of shape NxD

    Returns:
        maximum score array of shape N
    """
    return np.max(softmax, axis=1)


@register_confid_func("mcd_mcp")
@validate_softmax
@register_confid_func("mcd_mls")
def mcd_maximum_softmax_probability(
    mcd_softmax_mean: ArrayType, _: ArrayType
) -> ArrayType:
    """Maximum softmax probability CSF/Maximum logit score CSF for MCD

    Args:
        mcd_softmax_mean (ArrayType): array-like containing mean softmaxes or logits of shape NxD

    Returns:
        maximum score array of shape N
    """
    return np.max(mcd_softmax_mean, axis=1)


@register_confid_func("det_pe")
@validate_softmax
def predictive_entropy(softmax: ArrayType) -> ArrayType:
    """Predictive entropy CSF

    Args:
        softmax (ArrayType): array-like containing softmaxes of shape NxD

    Returns:
        pe score array of shape N
    """
    return np.sum(softmax * (-np.log(softmax + np.finfo(softmax.dtype).eps)), axis=1)


@register_confid_func("mcd_pe")
@validate_softmax
def mcd_predictive_entropy(mcd_softmax_mean: ArrayType, _: ArrayType) -> ArrayType:
    """Predictive entropy CSF for MCD

    Args:
        mcd_softmax_mean (ArrayType): array-like containing mean softmaxes of shape NxD

    Returns:
        pe score array of shape N
    """
    return predictive_entropy(mcd_softmax_mean)


@register_confid_func("mcd_ee")
@validate_softmax
def expected_entropy(_: ArrayType, mcd_softmax_dist: ArrayType) -> ArrayType:
    """Expected entropy over MCD data CSF

    Args:
        mcd_softmax_dist (ArrayType): array-like containing distribution of softmaxes of shape NxMxD

    Returns:
        ee score array of shape N
    """
    return np.mean(
        np.sum(
            mcd_softmax_dist
            * (-np.log(mcd_softmax_dist + np.finfo(mcd_softmax_dist.dtype).eps)),
            axis=1,
        ),
        axis=1,
    )


@register_confid_func("mcd_mi")
def mutual_information(
    mcd_softmax_mean: ArrayType, mcd_softmax_dist: ArrayType
) -> ArrayType:
    """Mutual information over MCD data CSF

    Args:
        mcd_softmax_mean (ArrayType): array-like containing mean of softmaxes of shape NxD
        mcd_softmax_dist (ArrayType): array-like containing distribution of softmaxes of shape NxMxD

    Returns:
        mi score array of shape N
    """
    return predictive_entropy(mcd_softmax_mean) - expected_entropy(
        mcd_softmax_mean, mcd_softmax_dist
    )


@register_confid_func("mcd_sv")
@validate_softmax
def softmax_variance(_: ArrayType, mcd_softmax_dist: ArrayType) -> ArrayType:
    """Softmax variance over MCD data CSF

    Args:
        mcd_softmax_dist (ArrayType): array-like containing distribution of softmaxes of shape NxMxD

    Returns:
        sv score array of shape N
    """
    return np.mean(np.std(mcd_softmax_dist, axis=2), axis=(1))


@register_confid_func("mcd_waic")
def mcd_waic(mcd_softmax_mean: ArrayType, mcd_softmax_dist: ArrayType) -> ArrayType:
    return np.max(mcd_softmax_mean, axis=1) - np.take(
        np.std(mcd_softmax_dist, axis=2),
        np.argmax(mcd_softmax_mean, axis=1),
    )


@register_confid_func("ext_waic")
@register_confid_func("bpd_waic")
@register_confid_func("maha_waic")
@register_confid_func("tcp_waic")
@register_confid_func("dg_waic")
@register_confid_func("devries_waic")
def ext_waic(mcd_softmax_mean: ArrayType, mcd_softmax_dist: ArrayType) -> ArrayType:
    return mcd_softmax_mean - np.std(mcd_softmax_dist, axis=1)


@register_confid_func("ext_mcd")
@register_confid_func("bpd_mcd")
@register_confid_func("maha_mcd")
@register_confid_func("tcp_mcd")
@register_confid_func("dg_mcd")
@register_confid_func("devries_mcd")
def mcd_ext(mcd_softmax_mean: ArrayType, _: ArrayType) -> ArrayType:
    """Dummy function for consistent handling of already computed confidences

    Args:
        mcd_softmax_mean (ArrayType): array-like containing already computed confidences

    Returns:
        unchanged confidences
    """
    return mcd_softmax_mean


@register_confid_func("ext")
@register_confid_func("bpd")
@register_confid_func("maha")
@register_confid_func("maha_qt")
@register_confid_func("temp_logits")
@register_confid_func("ext_qt")
@register_confid_func("tcp")
@register_confid_func("dg")
@register_confid_func("devries")
def ext_confid(softmax: ArrayType) -> ArrayType:
    """Dummy function for consistent handling of already computed confidences

    Args:
        softmax (ArrayType): array-like containing already computed confidences

    Returns:
        unchanged confidences
    """
    return softmax


class ConfidScore:
    """Wrapper class handling CSF selection and handing over function args

    Attributes:
        confid_func:
        analysis:
    """

    def __init__(
        self,
        study_data: ExperimentData,
        query_confid: str,
        analysis: Analysis,
    ) -> None:
        if is_mcd_confid(query_confid):
            assert study_data.mcd_softmax_mean is not None
            assert study_data.mcd_softmax_dist is not None
            self.softmax = study_data.mcd_softmax_mean
            self.correct = study_data.mcd_correct
            self.labels = study_data.mcd_labels

            self.confid_args = (
                study_data.mcd_softmax_mean,
                study_data.mcd_softmax_dist,
            )
            self.performance_args = (
                study_data.mcd_softmax_mean,
                study_data.mcd_labels,
                study_data.mcd_correct,
            )

            if is_external_confid(query_confid):
                assert study_data.mcd_external_confids_dist is not None
                self.confid_args = (
                    np.mean(study_data.mcd_external_confids_dist, axis=1),
                    study_data.mcd_external_confids_dist,
                )

            elif "mls" in query_confid:
                assert study_data.mcd_logits_dist is not None
                self.confid_args = (
                    study_data.mcd_logits_mean,
                    study_data.mcd_logits_dist,
                )

        else:
            self.softmax = study_data.softmax_output
            self.correct = study_data.correct
            self.labels = study_data.labels
            self.confid_args = (study_data.softmax_output,)
            self.performance_args = (
                study_data.softmax_output,
                study_data.labels,
                study_data.correct,
            )

            if is_external_confid(query_confid):
                assert study_data.external_confids is not None
                self.confid_args = (study_data.external_confids,)

            elif "mls" in query_confid:
                assert study_data.logits is not None
                self.confid_args = (study_data.logits,)

        self.confid_func = get_confid_function(query_confid)
        self.analysis = analysis

    @property
    def confids(self) -> ArrayType:
        return self.confid_func(*self.confid_args)

    @property
    def predict(self) -> ArrayType:
        return np.argmax(self.softmax, axis=1)

    @property
    def metrics(self) -> dict[Any, Any]:
        return self.analysis._compute_performance_metrics(*self.performance_args)


_combine_opts = {
    "average": lambda x, y: (x + y) / 2,
    "product": lambda x, y: x * y,
}


def parse_secondary_confid(query_confid: str, analysis: Analysis):
    parts = query_confid.split("-")
    assert len(parts) == 3, "Invalid secondary confid definition"

    return (
        (
            analysis.method_dict[parts[0]]["confids"],
            analysis.method_dict[parts[1]]["confids"],
        ),
        _combine_opts[parts[2]],
    )


class SecondaryConfidScore:
    """Wrapper class handling CSF selection and handing over function args for CSFs computed on top
    of precomputed confidence scores

    Attributes:
        confid_func:
        analysis:
    """

    def __init__(
        self, study_data: ExperimentData, query_confid: str, analysis: Analysis
    ):
        self.softmax = study_data.softmax_output
        self.correct = study_data.correct
        self.performance_args = (
            study_data.softmax_output,
            study_data.labels,
            study_data.correct,
        )
        self.labels = study_data.labels

        self.confid_args, self.confid_func = parse_secondary_confid(
            query_confid, analysis
        )
        self.analysis = analysis

    @property
    def confids(self) -> ArrayType:
        return self.confid_func(*self.confid_args)

    @property
    def predict(self) -> ArrayType:
        return np.argmax(self.softmax, axis=1)

    @property
    def metrics(self) -> dict[Any, Any]:
        return self.analysis._compute_performance_metrics(*self.performance_args)
