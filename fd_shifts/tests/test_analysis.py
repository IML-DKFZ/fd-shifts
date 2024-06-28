from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.metrics import roc_auc_score

from fd_shifts.analysis import metrics
from fd_shifts.analysis.rc_stats import RiskCoverageStatsMixin
from fd_shifts.tests.utils import (
    RC_STATS_CLASS_AWARE_TEST_CASES,
    RC_STATS_TEST_CASES,
    SC_scale1000_test,
    SC_test,
)

ArrayType = npt.NDArray[np.floating]
ExpectedType = float | ArrayType | type[BaseException]

N_SAMPLES = 100
N_BINS = 20

all_correct = np.ones(N_SAMPLES)
some_correct = np.arange(N_SAMPLES) % 2
none_correct = np.zeros(N_SAMPLES)

index = (np.arange(N_SAMPLES) % 3) > 0
all_nan = np.full(N_SAMPLES, np.nan)
some_nan_correct = np.where(index, some_correct, np.nan)

all_confid = np.ones(N_SAMPLES)
med_confid = np.arange(10).repeat(N_SAMPLES // 10) * 0.1
some_confid = np.arange(N_SAMPLES) % 2
some_confid_inv = (np.arange(N_SAMPLES) + 1) % 2
none_confid = np.zeros(N_SAMPLES)
unnormalized_confid = np.arange(-(N_SAMPLES // 2), N_SAMPLES // 2)

some_nan_confid = np.where(index, some_confid, np.nan)


stats_caches = {
    "all_correct": {
        "all_confid": SC_scale1000_test(all_confid, all_correct),
        "some_confid": SC_scale1000_test(some_confid, all_correct),
        "some_confid_inv": SC_scale1000_test(some_confid_inv, all_correct),
        "med_confid": SC_scale1000_test(med_confid, all_correct),
        "none_confid": SC_scale1000_test(none_confid, all_correct),
        "unnormalized_confid": SC_scale1000_test(unnormalized_confid, all_correct),
        "all_nan_confid": SC_scale1000_test(all_nan, all_correct),
        "some_nan_confid": SC_scale1000_test(some_nan_confid, all_correct),
    },
    "some_correct": {
        "all_confid": SC_scale1000_test(all_confid, some_correct),
        "some_confid": SC_scale1000_test(some_confid, some_correct),
        "some_confid_inv": SC_scale1000_test(some_confid_inv, some_correct),
        "med_confid": SC_scale1000_test(med_confid, some_correct),
        "none_confid": SC_scale1000_test(none_confid, some_correct),
        "unnormalized_confid": SC_scale1000_test(unnormalized_confid, some_correct),
        "all_nan_confid": SC_scale1000_test(all_nan, some_correct),
        "some_nan_confid": SC_scale1000_test(some_nan_confid, some_correct),
    },
    "none_correct": {
        "all_confid": SC_scale1000_test(all_confid, none_correct),
        "some_confid": SC_scale1000_test(some_confid, none_correct),
        "some_confid_inv": SC_scale1000_test(some_confid_inv, none_correct),
        "med_confid": SC_scale1000_test(med_confid, none_correct),
        "none_confid": SC_scale1000_test(none_confid, none_correct),
        "unnormalized_confid": SC_scale1000_test(unnormalized_confid, none_correct),
        "all_nan_confid": SC_scale1000_test(all_nan, none_correct),
        "some_nan_confid": SC_scale1000_test(some_nan_confid, none_correct),
    },
    "all_nan_correct": {
        "all_confid": SC_scale1000_test(all_confid, all_nan),
        "some_confid": SC_scale1000_test(some_confid, all_nan),
        "some_confid_inv": SC_scale1000_test(some_confid_inv, all_nan),
        "med_confid": SC_scale1000_test(med_confid, all_nan),
        "none_confid": SC_scale1000_test(none_confid, all_nan),
        "unnormalized_confid": SC_scale1000_test(unnormalized_confid, all_nan),
        "all_nan_confid": SC_scale1000_test(all_nan, all_nan),
        "some_nan_confid": SC_scale1000_test(some_nan_confid, all_nan),
    },
    "some_nan_correct": {
        "all_confid": SC_scale1000_test(all_confid, some_nan_correct),
        "some_confid": SC_scale1000_test(some_confid, some_nan_correct),
        "some_confid_inv": SC_scale1000_test(some_confid_inv, some_nan_correct),
        "med_confid": SC_scale1000_test(med_confid, some_nan_correct),
        "none_confid": SC_scale1000_test(none_confid, some_nan_correct),
        "unnormalized_confid": SC_scale1000_test(unnormalized_confid, some_nan_correct),
        "all_nan_confid": SC_scale1000_test(all_nan, some_nan_correct),
        "some_nan_confid": SC_scale1000_test(some_nan_confid, some_nan_correct),
    },
}


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], np.nan),
        (stats_caches["all_correct"]["some_confid"], np.nan),
        (stats_caches["all_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_correct"]["med_confid"], np.nan),
        (stats_caches["all_correct"]["none_confid"], np.nan),
        (stats_caches["all_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_correct"]["all_confid"], 0.5),
        (stats_caches["some_correct"]["some_confid"], 1),
        (stats_caches["some_correct"]["some_confid_inv"], 0),
        (stats_caches["some_correct"]["med_confid"], 0.5),
        (stats_caches["some_correct"]["none_confid"], 0.5),
        (stats_caches["some_correct"]["unnormalized_confid"], 0.51),
        (stats_caches["some_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_correct"]["some_nan_confid"], np.nan),
        (stats_caches["none_correct"]["all_confid"], np.nan),
        (stats_caches["none_correct"]["some_confid"], np.nan),
        (stats_caches["none_correct"]["some_confid_inv"], np.nan),
        (stats_caches["none_correct"]["med_confid"], np.nan),
        (stats_caches["none_correct"]["none_confid"], np.nan),
        (stats_caches["none_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["none_correct"]["all_nan_confid"], np.nan),
        (stats_caches["none_correct"]["some_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_failauc(stats_cache: SC_scale1000_test, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.failauc(stats_cache)

        return

    score = metrics.failauc(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], np.nan),
        (stats_caches["all_correct"]["some_confid"], np.nan),
        (stats_caches["all_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_correct"]["med_confid"], np.nan),
        (stats_caches["all_correct"]["none_confid"], np.nan),
        (stats_caches["all_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_correct"]["all_confid"], 1),
        (stats_caches["some_correct"]["some_confid"], 0),
        (stats_caches["some_correct"]["some_confid_inv"], 1),
        (stats_caches["some_correct"]["med_confid"], 1),
        (stats_caches["some_correct"]["none_confid"], 1),
        (stats_caches["some_correct"]["unnormalized_confid"], 0.94),
        (stats_caches["some_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_correct"]["some_nan_confid"], np.nan),
        (stats_caches["none_correct"]["all_confid"], np.nan),
        (stats_caches["none_correct"]["some_confid"], np.nan),
        (stats_caches["none_correct"]["some_confid_inv"], np.nan),
        (stats_caches["none_correct"]["med_confid"], np.nan),
        (stats_caches["none_correct"]["none_confid"], np.nan),
        (stats_caches["none_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["none_correct"]["all_nan_confid"], np.nan),
        (stats_caches["none_correct"]["some_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_fpr_at_95_tpr(stats_cache: SC_scale1000_test, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.fpr_at_95_tpr(stats_cache)

        return

    score = metrics.fpr_at_95_tpr(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], 1),
        (stats_caches["all_correct"]["some_confid"], 1),
        (stats_caches["all_correct"]["some_confid_inv"], 1),
        (stats_caches["all_correct"]["med_confid"], 1),
        (stats_caches["all_correct"]["none_confid"], 1),
        (stats_caches["all_correct"]["unnormalized_confid"], 1),
        (stats_caches["all_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_correct"]["all_confid"], 0.5),
        (stats_caches["some_correct"]["some_confid"], 1),
        (stats_caches["some_correct"]["some_confid_inv"], 0.5),
        (stats_caches["some_correct"]["med_confid"], 0.5),
        (stats_caches["some_correct"]["none_confid"], 0.5),
        (stats_caches["some_correct"]["unnormalized_confid"], 0.5293777484847491),
        (stats_caches["some_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_correct"]["some_nan_confid"], np.nan),
        (stats_caches["none_correct"]["all_confid"], 0),
        (stats_caches["none_correct"]["some_confid"], 0),
        (stats_caches["none_correct"]["some_confid_inv"], 0),
        (stats_caches["none_correct"]["med_confid"], 0),
        (stats_caches["none_correct"]["none_confid"], 0),
        (stats_caches["none_correct"]["unnormalized_confid"], 0),
        (stats_caches["none_correct"]["all_nan_confid"], np.nan),
        (stats_caches["none_correct"]["some_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_failap_suc(stats_cache: SC_scale1000_test, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.failap_suc(stats_cache)

        return

    score = metrics.failap_suc(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], 0),
        (stats_caches["all_correct"]["some_confid"], 0),
        (stats_caches["all_correct"]["some_confid_inv"], 0),
        (stats_caches["all_correct"]["med_confid"], 0),
        (stats_caches["all_correct"]["none_confid"], 0),
        (stats_caches["all_correct"]["unnormalized_confid"], 0),
        (stats_caches["all_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_correct"]["all_confid"], 0.5),
        (stats_caches["some_correct"]["some_confid"], 1),
        (stats_caches["some_correct"]["some_confid_inv"], 0.5),
        (stats_caches["some_correct"]["med_confid"], 0.5),
        (stats_caches["some_correct"]["none_confid"], 0.5),
        (stats_caches["some_correct"]["unnormalized_confid"], 0.5293777484847491),
        (stats_caches["some_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_correct"]["some_nan_confid"], np.nan),
        (stats_caches["none_correct"]["all_confid"], 1),
        (stats_caches["none_correct"]["some_confid"], 1),
        (stats_caches["none_correct"]["some_confid_inv"], 1),
        (stats_caches["none_correct"]["med_confid"], 1),
        (stats_caches["none_correct"]["none_confid"], 1),
        (stats_caches["none_correct"]["unnormalized_confid"], 1),
        (stats_caches["none_correct"]["all_nan_confid"], np.nan),
        (stats_caches["none_correct"]["some_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_failap_err(stats_cache: SC_scale1000_test, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.failap_err(stats_cache)

        return

    score = metrics.failap_err(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.legacy
@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], 0),
        (stats_caches["all_correct"]["some_confid"], 0),
        (stats_caches["all_correct"]["some_confid_inv"], 0),
        (stats_caches["all_correct"]["med_confid"], 0),
        (stats_caches["all_correct"]["none_confid"], 0),
        (stats_caches["all_correct"]["unnormalized_confid"], 0),
        (stats_caches["some_correct"]["all_confid"], 490.02525252525254),
        (stats_caches["some_correct"]["some_confid"], 128.71212121212122),
        (stats_caches["some_correct"]["some_confid_inv"], 861.2878787878789),
        (stats_caches["some_correct"]["med_confid"], 488.47515054956466),
        (stats_caches["some_correct"]["none_confid"], 490.02525252525254),
        (stats_caches["some_correct"]["unnormalized_confid"], 482.81112575762546),
        (stats_caches["none_correct"]["all_confid"], 990),
        (stats_caches["none_correct"]["some_confid"], 990),
        (stats_caches["none_correct"]["some_confid_inv"], 990),
        (stats_caches["none_correct"]["med_confid"], 990),
        (stats_caches["none_correct"]["none_confid"], 990),
        (stats_caches["none_correct"]["unnormalized_confid"], 990),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_aurc(stats_cache: SC_scale1000_test, expected: ExpectedType):
    stats_cache.legacy = True

    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.aurc(stats_cache)

        return

    score = metrics.aurc(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.legacy
@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], 0),
        (stats_caches["all_correct"]["some_confid"], 0),
        (stats_caches["all_correct"]["some_confid_inv"], 0),
        (stats_caches["all_correct"]["med_confid"], 0),
        (stats_caches["all_correct"]["none_confid"], 0),
        (stats_caches["all_correct"]["unnormalized_confid"], 0),
        (stats_caches["all_correct"]["all_nan_confid"], 0),
        (stats_caches["all_correct"]["some_nan_confid"], 0),
        (stats_caches["some_correct"]["all_confid"], 336.598842805225),
        (stats_caches["some_correct"]["some_confid"], -24.71428850790636),
        (stats_caches["some_correct"]["some_confid_inv"], 707.8614690678513),
        (stats_caches["some_correct"]["med_confid"], 335.0487408295371),
        (stats_caches["some_correct"]["none_confid"], 336.598842805225),
        (stats_caches["some_correct"]["unnormalized_confid"], 329.38471603759785),
        (stats_caches["some_correct"]["all_nan_confid"], 337.86666449164534),
        (stats_caches["some_correct"]["some_nan_confid"], 262.0011457609846),
        (stats_caches["none_correct"]["all_confid"], -10),
        (stats_caches["none_correct"]["some_confid"], -10),
        (stats_caches["none_correct"]["some_confid_inv"], -10),
        (stats_caches["none_correct"]["med_confid"], -10),
        (stats_caches["none_correct"]["none_confid"], -10),
        (stats_caches["none_correct"]["unnormalized_confid"], -10),
        (stats_caches["none_correct"]["all_nan_confid"], -10),
        (stats_caches["none_correct"]["some_nan_confid"], -10),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_eaurc(stats_cache: SC_scale1000_test, expected: ExpectedType):
    stats_cache.legacy = True

    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.eaurc(stats_cache)

        return

    score = metrics.eaurc(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], 1),
        (stats_caches["all_correct"]["some_confid"], 1),
        (stats_caches["all_correct"]["some_confid_inv"], 1),
        (stats_caches["all_correct"]["med_confid"], 0.9),
        (stats_caches["all_correct"]["none_confid"], 0),
        (stats_caches["all_correct"]["unnormalized_confid"], 1),
        (stats_caches["all_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_correct"]["all_confid"], 0.5),
        (stats_caches["some_correct"]["some_confid"], 0),
        (stats_caches["some_correct"]["some_confid_inv"], 1),
        (stats_caches["some_correct"]["med_confid"], 0.5),
        (stats_caches["some_correct"]["none_confid"], 0.5),
        (stats_caches["some_correct"]["unnormalized_confid"], 0.49019607843137253),
        (stats_caches["some_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_correct"]["some_nan_confid"], np.nan),
        (stats_caches["none_correct"]["all_confid"], 1),
        (stats_caches["none_correct"]["some_confid"], 1),
        (stats_caches["none_correct"]["some_confid_inv"], 1),
        (stats_caches["none_correct"]["med_confid"], 0.9),
        (stats_caches["none_correct"]["none_confid"], 0),
        (stats_caches["none_correct"]["unnormalized_confid"], 1),
        (stats_caches["none_correct"]["all_nan_confid"], np.nan),
        (stats_caches["none_correct"]["some_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_maximum_calibration_error(
    stats_cache: SC_scale1000_test, expected: ExpectedType
):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.maximum_calibration_error(stats_cache)

        return

    score = metrics.maximum_calibration_error(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], 1),
        (stats_caches["all_correct"]["some_confid"], 0.5),
        (stats_caches["all_correct"]["some_confid_inv"], 0.5),
        (stats_caches["all_correct"]["med_confid"], 0.45),
        (stats_caches["all_correct"]["none_confid"], 0),
        (stats_caches["all_correct"]["unnormalized_confid"], 0.49),
        (stats_caches["some_correct"]["all_confid"], 0.5),
        (stats_caches["some_correct"]["some_confid"], 0),
        (stats_caches["some_correct"]["some_confid_inv"], 1),
        (stats_caches["some_correct"]["med_confid"], 0.25),
        (stats_caches["some_correct"]["none_confid"], 0.5),
        (stats_caches["some_correct"]["unnormalized_confid"], 0.49),
        (stats_caches["none_correct"]["all_confid"], 1),
        (stats_caches["none_correct"]["some_confid"], 0.5),
        (stats_caches["none_correct"]["some_confid_inv"], 0.5),
        (stats_caches["none_correct"]["med_confid"], 0.45),
        (stats_caches["none_correct"]["none_confid"], 0),
        (stats_caches["none_correct"]["unnormalized_confid"], 0.49),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_expected_calibration_error(
    stats_cache: SC_scale1000_test, expected: ExpectedType
):
    """See reference
    https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/stats/calibration.py#L258-L319
    """
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.expected_calibration_error(stats_cache)

        return

    score = metrics.expected_calibration_error(stats_cache)
    np.testing.assert_almost_equal(score, expected)


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_caches["all_correct"]["all_confid"], 0),
        (stats_caches["all_correct"]["some_confid"], 8.059047775479163),
        (stats_caches["all_correct"]["some_confid_inv"], 8.059047775479163),
        (stats_caches["all_correct"]["med_confid"], 2.4039531178855778),
        (stats_caches["all_correct"]["none_confid"], 16.11809565095832),
        (stats_caches["all_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_correct"]["all_confid"], 8.059047775479163),
        (stats_caches["some_correct"]["some_confid"], 0),
        (stats_caches["some_correct"]["some_confid_inv"], 16.11809565095832),
        (stats_caches["some_correct"]["med_confid"], 1.5980483303376622),
        (stats_caches["some_correct"]["none_confid"], 8.059047775479163),
        (stats_caches["some_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_correct"]["some_nan_confid"], np.nan),
        (stats_caches["none_correct"]["all_confid"], 16.11809565095832),
        (stats_caches["none_correct"]["some_confid"], 8.059047775479163),
        (stats_caches["none_correct"]["some_confid_inv"], 8.059047775479163),
        (stats_caches["none_correct"]["med_confid"], 0.7921435427897465),
        (stats_caches["none_correct"]["none_confid"], 0),
        (stats_caches["none_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["none_correct"]["all_nan_confid"], np.nan),
        (stats_caches["none_correct"]["some_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["all_nan_correct"]["med_confid"], np.nan),
        (stats_caches["all_nan_correct"]["none_confid"], np.nan),
        (stats_caches["all_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["all_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["all_nan_correct"]["some_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_confid_inv"], np.nan),
        (stats_caches["some_nan_correct"]["med_confid"], np.nan),
        (stats_caches["some_nan_correct"]["none_confid"], np.nan),
        (stats_caches["some_nan_correct"]["unnormalized_confid"], np.nan),
        (stats_caches["some_nan_correct"]["all_nan_confid"], np.nan),
        (stats_caches["some_nan_correct"]["some_nan_confid"], np.nan),
    ],
)
def test_failnll(stats_cache: SC_scale1000_test, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.failnll(stats_cache)

        return

    score = metrics.failnll(stats_cache)
    np.testing.assert_almost_equal(score, expected)


def test_caching():
    """Test property caching"""
    stats_cache = SC_test(confids=np.ones(3), correct=np.linspace(0, 1, 3))
    np.testing.assert_almost_equal(stats_cache.aurc, 0.5)
    np.testing.assert_equal(stats_cache.dominant_point_mask, [True, False])
    # Corrupt underlying data, properties should still be cached
    stats_cache.correct = None
    np.testing.assert_almost_equal(stats_cache.aurc, 0.5)
    np.testing.assert_equal(stats_cache.dominant_point_mask, [True, False])


def test_rcs_validation():
    """Test RiskCoverageStats data validation"""

    class MissingCorrect(RiskCoverageStatsMixin):
        def __init__(self, confids):
            super().__init__()
            self.confids = confids

    class MissingConfids(RiskCoverageStatsMixin):
        def __init__(self, residuals):
            super().__init__()
            self.residuals = residuals

    with pytest.raises(AssertionError, match="Missing class member 'residuals'"):
        MissingCorrect(confids=np.ones(3))._validate()
    with pytest.raises(AssertionError, match="Missing class member 'confids'"):
        MissingConfids(residuals=np.ones(3))._validate()

    stats_cache = SC_test(confids=np.ones(3), correct=np.ones(3) * 2)
    with pytest.raises(
        ValueError, match="Must provide either target_risk or target_cov value"
    ):
        stats_cache.get_working_point(risk="selective-risk")
    with pytest.raises(ValueError, match="arguments are mutually exclusive"):
        stats_cache.get_working_point(
            risk="selective-risk", target_risk=0.0, target_cov=0.0
        )


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [(stats_cache, exp) for stats_cache, exp in RC_STATS_TEST_CASES.items()],
    # ID for identifying failing test cases
    ids=[exp["ID"] for exp in RC_STATS_TEST_CASES.values()],
)
def test_class_agnostic_metric_values(stats_cache: SC_test, expected: dict):
    """"""
    if stats_cache.contains_nan:
        assert np.isnan(stats_cache.aurc)
        assert np.isnan(stats_cache.eaurc)
        assert np.isnan(stats_cache.aurc_achievable)
        assert np.isnan(stats_cache.eaurc_achievable)
        assert np.isnan(stats_cache.augrc)
        assert np.isnan(stats_cache.aurc_ba)
        assert np.isnan(stats_cache.augrc_ba)
        assert all(
            np.isnan(
                stats_cache.get_working_point(risk="selective-risk", target_cov=0.5)
            )
        )

    else:
        # Now, compare to explicit result values
        np.testing.assert_almost_equal(stats_cache.aurc, expected["aurc"])
        np.testing.assert_almost_equal(stats_cache.eaurc, expected["eaurc"])
        # TODO
        np.testing.assert_almost_equal(
            stats_cache.aurc_achievable, expected["aurc_achievable"]
        )
        np.testing.assert_almost_equal(
            stats_cache.eaurc_achievable, expected["eaurc_achievable"]
        )
        np.testing.assert_almost_equal(
            stats_cache.get_working_point(risk="selective-risk", target_cov=0.5),
            expected["selective-risk@50cov"],
        )
        np.testing.assert_almost_equal(
            stats_cache.get_working_point(risk="selective-risk", target_cov=0.95),
            expected["selective-risk@95cov"],
        )
        np.testing.assert_almost_equal(stats_cache.augrc, expected["augrc"])

        # For binary residuals, test that AUGRC matches the theoretical result based on
        # failure-AUROC and accuracy
        if stats_cache.is_binary:
            acc = np.mean(stats_cache.correct)
            if acc == 1 or acc == 0:
                auroc = 1
            else:
                auroc = roc_auc_score(stats_cache.correct, stats_cache.confids)
            np.testing.assert_almost_equal(
                stats_cache.augrc, (1 - auroc) * acc * (1 - acc) + 0.5 * (1 - acc) ** 2
            )


@pytest.mark.parametrize(
    ("stats_cache", "expected"),
    [
        (stats_cache, exp)
        for stats_cache, exp in RC_STATS_CLASS_AWARE_TEST_CASES.items()
    ],
    # ID for identifying failing test cases
    ids=[exp["ID"] for exp in RC_STATS_CLASS_AWARE_TEST_CASES.values()],
)
def test_class_aware_metric_values(stats_cache: SC_test, expected: dict):
    """"""
    if stats_cache.contains_nan:
        assert np.isnan(stats_cache.aurc_ba)
        assert np.isnan(stats_cache.augrc_ba)
        assert all(
            np.isnan(
                stats_cache.get_working_point(
                    risk="generalized-risk-ba", target_cov=0.5
                )
            )
        )

    else:
        # Now, compare to explicit result values
        np.testing.assert_almost_equal(stats_cache.aurc_ba, expected["aurc_ba"])
        np.testing.assert_almost_equal(stats_cache.augrc_ba, expected["augrc_ba"])


def test_achievable_rc():
    """Test toy examples for achievable AURC, e-AURC, and dominant point masks"""
    confids = np.array([0.0, 0.0, 0.2, 0.4, 0.6, 0.8])
    correct = np.array([1, 0, 1, 0, 1, 1])
    rcs = SC_test(confids, correct)
    assert len(rcs.coverages) == 6
    np.testing.assert_equal(rcs.thresholds, [0.0, 0.2, 0.4, 0.6, 0.8])
    np.testing.assert_equal(
        rcs.dominant_point_mask, [True, False, False, True, False, False]
    )
    assert rcs.aurc_achievable < rcs.aurc
    assert rcs.eaurc_achievable < rcs.eaurc

    confids = np.repeat([0.2, 0.4, 0.6, 0.8], 25)
    correct = np.clip(np.linspace(-1, 1, 100), 0, 1)
    rcs = SC_test(confids=confids, correct=correct)
    np.testing.assert_almost_equal(
        rcs.evaluate_auc(risk="selective-risk", interpolation="non-linear")
        - rcs.aurc_optimal,
        rcs.eaurc_achievable,
    )


def test_aurc_coverage_range():
    """Test AURC evaluation over a certain coverage range"""
    from itertools import product

    # Most simple case: Equal confidences
    rcs = SC_test(confids=np.ones(4), correct=np.array([0, 1, 1, 1]))
    np.testing.assert_almost_equal(rcs.aurc, 0.25)

    for lower, upper in product((0.0, 0.3, 0.6, 1.0), (0.4, 0.7, 1.0)):
        np.testing.assert_almost_equal(
            rcs.evaluate_auc(
                risk="selective-risk",
                cov_min=lower,
                cov_max=upper,
                interpolation="non-linear",
            ),
            rcs.aurc * max((upper - lower), 0.0),
        )

    # Stratified confidences
    rcs = SC_test(confids=np.linspace(0, 1, 5), correct=np.array([0, 0, 1, 1, 1]))
    for lower in np.linspace(0.0, 0.6, 10):
        np.testing.assert_almost_equal(
            rcs.evaluate_auc(
                risk="selective-risk",
                cov_min=lower,
                interpolation="non-linear",
            ),
            rcs.evaluate_auc(
                risk="selective-risk",
                interpolation="non-linear",
            ),
        )

    # assert that segments add up to complete AURC
    np.testing.assert_almost_equal(
        sum(
            [
                rcs.evaluate_auc(
                    risk="selective-risk",
                    cov_min=lower,
                    cov_max=upper,
                    interpolation="non-linear",
                )
                for lower, upper in zip(rcs.coverages[1:], rcs.coverages[:-1])
            ]
            + [
                rcs.evaluate_auc(
                    risk="selective-risk",
                    cov_max=rcs.coverages[-1],
                    interpolation="non-linear",
                )
            ]
        ),
        rcs.evaluate_auc(
            risk="selective-risk",
            interpolation="non-linear",
        ),
    )

    # Same test for random and more segments
    rcs = SC_test(
        confids=np.random.random(size=100), correct=np.random.random(size=100)
    )
    np.testing.assert_almost_equal(
        sum(
            [
                rcs.evaluate_auc(
                    risk="selective-risk",
                    cov_min=lower,
                    cov_max=upper,
                    interpolation="non-linear",
                )
                for lower, upper in zip(rcs.coverages[1:], rcs.coverages[:-1])
            ]
            + [
                rcs.evaluate_auc(
                    risk="selective-risk",
                    cov_max=rcs.coverages[-1],
                    interpolation="non-linear",
                )
            ]
        ),
        rcs.evaluate_auc(
            risk="selective-risk",
            interpolation="non-linear",
        ),
    )
