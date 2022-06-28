from typing import Any, Type, Union

import numpy as np
import numpy.typing as npt
import pytest
import scipy.special as sp

from fd_shifts.analysis import confid_scores

N_SAMPLES = 100
N_CLASSES = 10
N_MCD_SAMPLES = 3

# All classes have equal probability
_softmax_const64 = sp.softmax(np.ones((N_SAMPLES, N_CLASSES), dtype=np.float64), axis=1)
_softmax_const16 = sp.softmax(np.ones((N_SAMPLES, N_CLASSES), dtype=np.float16), axis=1)

# One class is highly likely, numerical issues in half precision
_softmax_extreme16 = np.array(
    [[*([1e-7] * N_CLASSES), 1 - ((N_CLASSES - 1) * 1e-7)]] * N_SAMPLES,
    dtype=np.float16,
)
_softmax_extreme64 = np.array(
    [[*([1e-7] * N_CLASSES), 1 - ((N_CLASSES - 1) * 1e-7)]] * N_SAMPLES,
    dtype=np.float64,
)

# All nan
_softmax_allnan64 = sp.softmax(
    np.full((N_SAMPLES, N_CLASSES), np.nan, dtype=np.float64), axis=1
)
_softmax_allnan16 = sp.softmax(
    np.full((N_SAMPLES, N_CLASSES), np.nan, dtype=np.float16), axis=1
)

# Some nan
index = (np.arange(N_CLASSES * N_SAMPLES) % (N_CLASSES - 1)).reshape(
    N_SAMPLES, N_CLASSES
) > 0
_softmax_somenan64 = np.where(index, _softmax_const64, np.nan)
_softmax_somenan16 = np.where(index, _softmax_const16, np.nan)

# All classes have equal probability
_mcd_softmax_const64 = sp.softmax(
    np.ones((N_SAMPLES, N_CLASSES, N_MCD_SAMPLES), dtype=np.float64), axis=1
)
_mcd_softmax_const16 = sp.softmax(
    np.ones((N_SAMPLES, N_CLASSES, N_MCD_SAMPLES), dtype=np.float16), axis=1
)

# One class is highly likely, numerical issues in half precision
_mcd_softmax_extreme16 = np.array(
    [[[*([1e-7] * N_CLASSES), 1 - ((N_CLASSES - 1) * 1e-7)]] * N_MCD_SAMPLES]
    * N_SAMPLES,
    dtype=np.float16,
).transpose(0, 2, 1)
_mcd_softmax_extreme64 = np.array(
    [[[*([1e-7] * N_CLASSES), 1 - ((N_CLASSES - 1) * 1e-7)]] * N_MCD_SAMPLES]
    * N_SAMPLES,
    dtype=np.float64,
).transpose(0, 2, 1)

# All nan
_mcd_softmax_allnan64 = sp.softmax(
    np.full((N_SAMPLES, N_CLASSES), np.nan, dtype=np.float64), axis=1
)
_mcd_softmax_allnan16 = sp.softmax(
    np.full((N_SAMPLES, N_CLASSES), np.nan, dtype=np.float16), axis=1
)

# Some nan
mcd_index = np.tile(index[:, :, None], (1, 1, N_MCD_SAMPLES))
_mcd_softmax_somenan64 = np.where(mcd_index, _mcd_softmax_const64, np.nan)
_mcd_softmax_somenan16 = np.where(mcd_index, _mcd_softmax_const16, np.nan)
assert _mcd_softmax_somenan16.shape == (N_SAMPLES, N_CLASSES, N_MCD_SAMPLES)
assert _mcd_softmax_somenan64.shape == (N_SAMPLES, N_CLASSES, N_MCD_SAMPLES)

# Some variance
# TODO: Maybe create something more different
_mcd_softmax_variant64 = (1 - (np.arange(N_MCD_SAMPLES, dtype=np.float64) % 3)) * np.array(
    [[[*([1e-4] * (N_CLASSES - 1)), -(N_CLASSES - 1) * 1e-4]] * N_MCD_SAMPLES]
    * N_SAMPLES,
    dtype=np.float64,
).transpose(0, 2, 1) + _mcd_softmax_const64
np.testing.assert_array_equal(_mcd_softmax_variant64.mean(axis=2), _softmax_const64)

_mcd_softmax_variant16 = (1 - (np.arange(N_MCD_SAMPLES, dtype=np.float16) % 3)) * np.array(
    [[[*([1e-4] * (N_CLASSES - 1)), -(N_CLASSES - 1) * 1e-4]] * N_MCD_SAMPLES]
    * N_SAMPLES,
    dtype=np.float16,
).transpose(0, 2, 1) + _mcd_softmax_const16
np.testing.assert_array_equal(_mcd_softmax_variant16.mean(axis=2), _softmax_const16)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_softmax_const16, 1 / N_CLASSES),
        (_softmax_const64, 1 / N_CLASSES),
        (_softmax_extreme16, AssertionError),
        (_softmax_extreme64, 1 - ((N_CLASSES - 1) * 1e-7)),
        (_softmax_somenan16, AssertionError),
        (_softmax_somenan64, AssertionError),
        (_softmax_allnan16, AssertionError),
        (_softmax_allnan64, AssertionError),
    ],
)
def test_maximum_softmax_probability(
    data: npt.NDArray[Any], expected: float | npt.NDArray[Any] | type[BaseException]
):
    # return np.max(softmax, axis=1)
    if isinstance(expected, type):
        with pytest.raises(expected):
            confid_scores.maximum_softmax_probability(data)

        return

    score = confid_scores.maximum_softmax_probability(data)
    np.testing.assert_array_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_softmax_const16, 2.293),
        (_softmax_const64, 2.30258509),
        (_softmax_extreme16, AssertionError),
        (_softmax_extreme64, 1.70181e-05),
        (_softmax_somenan16, AssertionError),
        (_softmax_somenan64, AssertionError),
        (_softmax_allnan16, AssertionError),
        (_softmax_allnan64, AssertionError),
    ],
)
def test_predictive_entropy(
    data: npt.NDArray[Any], expected: float | npt.NDArray[Any] | type[BaseException]
):
    # return np.sum(softmax * (-np.log(softmax + np.finfo(softmax.dtype).eps)), axis=1)
    if isinstance(expected, type):
        with pytest.raises(expected):
            confid_scores.predictive_entropy(data)

        return

    score = confid_scores.predictive_entropy(data)
    np.testing.assert_array_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("mcd_softmax_dist", "mcd_softmax_mean", "expected"),
    [
        (_mcd_softmax_const16, _softmax_const16, 2.293),
        (_mcd_softmax_const64, _softmax_const64, 2.30258509),

        (_mcd_softmax_extreme16, _softmax_extreme16, AssertionError),
        (_mcd_softmax_extreme64, _softmax_extreme64, 1.70181e-05),

        (_mcd_softmax_somenan16, _softmax_somenan16, AssertionError),
        (_mcd_softmax_somenan64, _softmax_somenan64, AssertionError),

        (_mcd_softmax_allnan16, _softmax_allnan16, AssertionError),
        (_mcd_softmax_allnan64, _softmax_allnan64, AssertionError),

        # NOTE: Did not know this would fail too, don't use half precision kids!
        (_mcd_softmax_variant16, _softmax_const16, AssertionError),
        (_mcd_softmax_variant64, _softmax_const64, 2.302582),
    ],
)
def test_expected_entropy(
    mcd_softmax_mean: npt.NDArray[Any],
    mcd_softmax_dist: npt.NDArray[Any],
    expected: float | npt.NDArray[Any] | type[BaseException],
):
    if isinstance(expected, type):
        with pytest.raises(expected):
            confid_scores.expected_entropy(mcd_softmax_mean, mcd_softmax_dist)

        return

    score = confid_scores.expected_entropy(mcd_softmax_mean, mcd_softmax_dist)
    np.testing.assert_array_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("mcd_softmax_dist", "mcd_softmax_mean", "expected"),
    [
        (_mcd_softmax_const16, _softmax_const16, 0),
        (_mcd_softmax_const64, _softmax_const64, 0),

        (_mcd_softmax_extreme16, _softmax_extreme16, AssertionError),
        (_mcd_softmax_extreme64, _softmax_extreme64, 0),

        (_mcd_softmax_somenan16, _softmax_somenan16, AssertionError),
        (_mcd_softmax_somenan64, _softmax_somenan64, AssertionError),

        (_mcd_softmax_allnan16, _softmax_allnan16, AssertionError),
        (_mcd_softmax_allnan64, _softmax_allnan64, AssertionError),

        (_mcd_softmax_variant16, _softmax_const16, AssertionError),
        (_mcd_softmax_variant64, _softmax_const64, 3.000037e-06),
    ],
)
def test_mutual_information(
    mcd_softmax_mean: npt.NDArray[Any],
    mcd_softmax_dist: npt.NDArray[Any],
    expected: float | npt.NDArray[Any] | type[BaseException],
):
    if isinstance(expected, type):
        with pytest.raises(expected):
            confid_scores.mutual_information(mcd_softmax_mean, mcd_softmax_dist)

        return

    score = confid_scores.mutual_information(mcd_softmax_mean, mcd_softmax_dist)
    np.testing.assert_array_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("mcd_softmax_dist", "mcd_softmax_mean", "expected"),
    [
        (_mcd_softmax_const16, _softmax_const16, 0),
        (_mcd_softmax_const64, _softmax_const64, 0),

        (_mcd_softmax_extreme16, _softmax_extreme16, AssertionError),
        (_mcd_softmax_extreme64, _softmax_extreme64, 0),

        (_mcd_softmax_somenan16, _softmax_somenan16, AssertionError),
        (_mcd_softmax_somenan64, _softmax_somenan64, AssertionError),

        (_mcd_softmax_allnan16, _softmax_allnan16, AssertionError),
        (_mcd_softmax_allnan64, _softmax_allnan64, AssertionError),

        (_mcd_softmax_variant16, _softmax_const16, AssertionError),
        (_mcd_softmax_variant64, _softmax_const64, 0.000147),
    ],
)
def test_softmax_variance(
    mcd_softmax_mean: npt.NDArray[Any],
    mcd_softmax_dist: npt.NDArray[Any],
    expected: float | npt.NDArray[Any] | type[BaseException],
):
    if isinstance(expected, type):
        with pytest.raises(expected):
            confid_scores.softmax_variance(mcd_softmax_mean, mcd_softmax_dist)

        return

    score = confid_scores.softmax_variance(mcd_softmax_mean, mcd_softmax_dist)
    np.testing.assert_array_almost_equal(score, expected)


def test_mcd_waic():
    # return np.max(mcd_softmax_mean, axis=1) - np.take(
    #     np.std(mcd_softmax_dist, axis=2),
    #     np.argmax(mcd_softmax_mean, axis=1),
    # )
    pass


def test_ext_waic():
    # return mcd_softmax_mean - np.std(mcd_softmax_dist, axis=1)
    pass
