from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from fd_shifts.analysis import metrics

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
        "all_confid": metrics.StatsCache(all_confid, all_correct, N_BINS),
        "some_confid": metrics.StatsCache(some_confid, all_correct, N_BINS),
        "some_confid_inv": metrics.StatsCache(some_confid_inv, all_correct, N_BINS),
        "med_confid": metrics.StatsCache(med_confid, all_correct, N_BINS),
        "none_confid": metrics.StatsCache(none_confid, all_correct, N_BINS),
        "unnormalized_confid": metrics.StatsCache(
            unnormalized_confid, all_correct, N_BINS
        ),
        "all_nan_confid": metrics.StatsCache(all_nan, all_correct, N_BINS),
        "some_nan_confid": metrics.StatsCache(some_nan_confid, all_correct, N_BINS),
    },
    "some_correct": {
        "all_confid": metrics.StatsCache(all_confid, some_correct, N_BINS),
        "some_confid": metrics.StatsCache(some_confid, some_correct, N_BINS),
        "some_confid_inv": metrics.StatsCache(some_confid_inv, some_correct, N_BINS),
        "med_confid": metrics.StatsCache(med_confid, some_correct, N_BINS),
        "none_confid": metrics.StatsCache(none_confid, some_correct, N_BINS),
        "unnormalized_confid": metrics.StatsCache(
            unnormalized_confid, some_correct, N_BINS
        ),
        "all_nan_confid": metrics.StatsCache(all_nan, some_correct, N_BINS),
        "some_nan_confid": metrics.StatsCache(some_nan_confid, some_correct, N_BINS),
    },
    "none_correct": {
        "all_confid": metrics.StatsCache(all_confid, none_correct, N_BINS),
        "some_confid": metrics.StatsCache(some_confid, none_correct, N_BINS),
        "some_confid_inv": metrics.StatsCache(some_confid_inv, none_correct, N_BINS),
        "med_confid": metrics.StatsCache(med_confid, none_correct, N_BINS),
        "none_confid": metrics.StatsCache(none_confid, none_correct, N_BINS),
        "unnormalized_confid": metrics.StatsCache(
            unnormalized_confid, none_correct, N_BINS
        ),
        "all_nan_confid": metrics.StatsCache(all_nan, none_correct, N_BINS),
        "some_nan_confid": metrics.StatsCache(some_nan_confid, none_correct, N_BINS),
    },
    "all_nan_correct": {
        "all_confid": metrics.StatsCache(all_confid, all_nan, N_BINS),
        "some_confid": metrics.StatsCache(some_confid, all_nan, N_BINS),
        "some_confid_inv": metrics.StatsCache(some_confid_inv, all_nan, N_BINS),
        "med_confid": metrics.StatsCache(med_confid, all_nan, N_BINS),
        "none_confid": metrics.StatsCache(none_confid, all_nan, N_BINS),
        "unnormalized_confid": metrics.StatsCache(unnormalized_confid, all_nan, N_BINS),
        "all_nan_confid": metrics.StatsCache(all_nan, all_nan, N_BINS),
        "some_nan_confid": metrics.StatsCache(some_nan_confid, all_nan, N_BINS),
    },
    "some_nan_correct": {
        "all_confid": metrics.StatsCache(all_confid, some_nan_correct, N_BINS),
        "some_confid": metrics.StatsCache(some_confid, some_nan_correct, N_BINS),
        "some_confid_inv": metrics.StatsCache(
            some_confid_inv, some_nan_correct, N_BINS
        ),
        "med_confid": metrics.StatsCache(med_confid, some_nan_correct, N_BINS),
        "none_confid": metrics.StatsCache(none_confid, some_nan_correct, N_BINS),
        "unnormalized_confid": metrics.StatsCache(
            unnormalized_confid, some_nan_correct, N_BINS
        ),
        "all_nan_confid": metrics.StatsCache(all_nan, some_nan_correct, N_BINS),
        "some_nan_confid": metrics.StatsCache(
            some_nan_confid, some_nan_correct, N_BINS
        ),
    },
}


def test_stats_cache():
    pass


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
def test_failauc(stats_cache: metrics.StatsCache, expected: ExpectedType):
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
def test_fpr_at_95_tpr(stats_cache: metrics.StatsCache, expected: ExpectedType):
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
def test_failap_suc(stats_cache: metrics.StatsCache, expected: ExpectedType):
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
def test_failap_err(stats_cache: metrics.StatsCache, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.failap_err(stats_cache)

        return

    score = metrics.failap_err(stats_cache)
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
def test_aurc(stats_cache: metrics.StatsCache, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.aurc(stats_cache)

        return

    score = metrics.aurc(stats_cache)
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
def test_eaurc(stats_cache: metrics.StatsCache, expected: ExpectedType):
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
    stats_cache: metrics.StatsCache, expected: ExpectedType
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
    stats_cache: metrics.StatsCache, expected: ExpectedType
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
def test_failnll(stats_cache: metrics.StatsCache, expected: ExpectedType):
    if isinstance(expected, type):
        with pytest.raises(expected):
            metrics.failnll(stats_cache)

        return

    score = metrics.failnll(stats_cache)
    np.testing.assert_almost_equal(score, expected)
