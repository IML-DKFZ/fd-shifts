from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from fd_shifts.analysis import Analysis, ExperimentData

EXTERNAL_CONFIDS = ["ext", "bpd", "maha", "tcp", "dg", "devries"]

# TODO: Add better error handling/reporting


def _check_shape_det(softmax: npt.NDArray[np.number]):
    assert len(softmax.shape) == 2


def _check_shape_mcd(softmax: npt.NDArray[np.number]):
    assert len(softmax.shape) == 3


def _assert_valid_softmax(softmax: npt.NDArray[np.number]):
    # TODO: Decide whether this should assert or propagate nan
    assert np.isfinite(softmax).all(), "NaN or INF in softmax output"
    np.testing.assert_array_almost_equal(softmax.sum(axis=1), 1.0)

    msr = softmax.max(axis=1)
    errors = ((msr == 1) & ((softmax > 0) & (softmax < 1)).any(axis=1))
    # TODO: Assert on acceptable error rate?
    assert not errors.any(), f"Numerical errors in softmax: {errors.mean() * 100:.2f}%"


def is_external_confid(name: str):
    return name.split("_")[0] in EXTERNAL_CONFIDS


def is_mcd_confid(name: str):
    return "mcd" in name or "waic" in name


_confid_funcs = {}


def register_confid_func(name: str) -> Callable:
    def _inner_wrapper(func: Callable) -> Callable:
        _confid_funcs[name] = func
        return func

    return _inner_wrapper


def confid_function_exists(confid_name):
    return confid_name in _confid_funcs


def get_confid_function(confid_name):
    if not confid_function_exists(confid_name):
        raise NotImplementedError(f"Function for {confid_name} not implemented.")

    return _confid_funcs[confid_name]


@register_confid_func("det_mcp")
@register_confid_func("det_mls")
def maximum_softmax_probability(softmax: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    _assert_valid_softmax(softmax)
    return np.max(softmax, axis=1)


@register_confid_func("mcd_mcp")
def mcd_maximum_softmax_probability(
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    return maximum_softmax_probability(mcd_softmax_mean)


@register_confid_func("det_pe")
def predictive_entropy(softmax: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    _assert_valid_softmax(softmax)
    return np.sum(softmax * (-np.log(softmax + np.finfo(softmax.dtype).eps)), axis=1)


@register_confid_func("mcd_pe")
def mcd_predictive_entropy(
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    return predictive_entropy(mcd_softmax_mean)


@register_confid_func("mcd_ee")
def expected_entropy(
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    _assert_valid_softmax(mcd_softmax_mean)
    _assert_valid_softmax(mcd_softmax_dist)
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
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    return predictive_entropy(mcd_softmax_mean) - expected_entropy(
        mcd_softmax_mean, mcd_softmax_dist
    )


@register_confid_func("mcd_sv")
def softmax_variance(
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    _assert_valid_softmax(mcd_softmax_mean)
    _assert_valid_softmax(mcd_softmax_dist)
    return np.mean(np.std(mcd_softmax_dist, axis=2), axis=(1))


@register_confid_func("mcd_waic")
def mcd_waic(
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    _assert_valid_softmax(mcd_softmax_mean)
    _assert_valid_softmax(mcd_softmax_dist)
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
def ext_waic(
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    _assert_valid_softmax(mcd_softmax_mean)
    _assert_valid_softmax(mcd_softmax_dist)
    return mcd_softmax_mean - np.std(mcd_softmax_dist, axis=1)


@register_confid_func("ext_mcd")
@register_confid_func("bpd_mcd")
@register_confid_func("maha_mcd")
@register_confid_func("tcp_mcd")
@register_confid_func("dg_mcd")
@register_confid_func("devries_mcd")
def mcd_ext(
    mcd_softmax_mean: npt.NDArray[np.number], mcd_softmax_dist: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    _assert_valid_softmax(mcd_softmax_mean)
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
def ext_confid(softmax: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    return softmax


class ConfidScore:
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
            self.confid_args = (
                study_data.mcd_softmax_mean,
                study_data.mcd_softmax_dist,
            )
            self.performance_args = (
                study_data.mcd_softmax_mean,
                study_data.labels,
                study_data.mcd_correct,
            )

            if is_external_confid(query_confid):
                assert study_data.mcd_external_confids_dist is not None
                self.confid_args = (
                    np.mean(study_data.mcd_external_confids_dist, axis=1),
                    study_data.mcd_external_confids_dist,
                )

        else:
            self.softmax = study_data.softmax_output
            self.correct = study_data.correct
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
    def confids(self) -> npt.NDArray[np.number]:
        return self.confid_func(*self.confid_args)

    @property
    def predict(self) -> npt.NDArray[np.number]:
        return np.argmax(self.softmax, axis=1)

    @property
    def metrics(self) -> dict[Any, Any]:
        return self.analysis.compute_performance_metrics(*self.performance_args)


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

        self.confid_args, self.confid_func = parse_secondary_confid(
            query_confid, analysis
        )
        self.analysis = analysis

    @property
    def confids(self) -> npt.NDArray[np.number]:
        return self.confid_func(*self.confid_args)

    @property
    def predict(self) -> npt.NDArray[np.number]:
        return np.argmax(self.softmax, axis=1)

    @property
    def metrics(self) -> dict[Any, Any]:
        return self.analysis.compute_performance_metrics(*self.performance_args)
