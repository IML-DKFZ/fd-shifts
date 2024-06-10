import os

import numpy as np
import pytest

from fd_shifts.analysis import metrics


@pytest.fixture
def mock_env_if_missing(monkeypatch) -> None:
    monkeypatch.setenv(
        "EXPERIMENT_ROOT_DIR", os.getenv("EXPERIMENT_ROOT_DIR", default="./experiments")
    )
    monkeypatch.setenv(
        "DATASET_ROOT_DIR", os.getenv("DATASET_ROOT_DIR", default="./data")
    )


class SC_test(metrics.StatsCache):
    """Using AURC_DISPLAY_SCALE=1 and n_bins=20 for testing."""

    AUC_DISPLAY_SCALE = 1

    def __init__(self, confids, correct):
        super().__init__(confids, correct, n_bins=20, legacy=False)


class SC_scale1000_test(metrics.StatsCache):
    """Using AURC_DISPLAY_SCALE=1000 and n_bins=20."""

    AUC_DISPLAY_SCALE = 1000

    def __init__(self, confids, correct):
        super().__init__(confids, correct, n_bins=20, legacy=False)


N_SAMPLES = 100
assert N_SAMPLES % 4 == 0
nan_index = (np.arange(N_SAMPLES) % 3) > 0

# confidence test cases
c_all_equal = np.ones(N_SAMPLES) * np.random.random()
c_alternating_01 = np.arange(N_SAMPLES) % 2
c_alternating_10 = (np.arange(N_SAMPLES) + 1) % 2
c_ascending = np.linspace(0.0, 1.0, N_SAMPLES)
c_ascending_unnormalized = np.arange(-(N_SAMPLES // 2), N_SAMPLES // 2)
c_some_nan = np.where(nan_index, c_alternating_01, np.nan)
c_plateaus = np.repeat([0.2, 0.4, 0.6, 0.8], N_SAMPLES // 4)

# prediction test cases
p_all_one = np.ones(N_SAMPLES)
p_all_zero = np.zeros(N_SAMPLES)
p_alternating_01 = np.arange(N_SAMPLES) % 2
p_some_nan = np.where(nan_index, p_alternating_01, np.nan)
p_plateau_permutation = np.concatenate(
    [
        np.random.permutation(
            p_alternating_01[i * N_SAMPLES // 4 : (i + 1) * N_SAMPLES // 4]
        )
        for i in range(4)
    ]
)
p_ascending = np.linspace(0.0, 1.0, N_SAMPLES)
p_plateaus = np.repeat([0.2, 0.4, 0.6, 0.8], N_SAMPLES // 4)

# =======================================================================================
# NOTE: For the following test cases, we use LINEAR interpolation of the Selective Risk
#       Coverage curve for the AURC computation and NON-LINEAR interpolation for the
#       achievable-AURC computation.
# =======================================================================================

# RiskCoverageStats test cases
RC_STATS_TEST_CASES = {
    # -- Confidence scores all the same -------------------------------------------------
    SC_test(c_all_equal, p_all_one): {
        "ID": "confid-all-equal_p-all-one",
        "aurc": 0.0,
        "eaurc": 0.0,
        "aurc_achievable": 0.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 0.0, c_all_equal[0]),
        "selective-risk@95cov": (1.0, 0.0, c_all_equal[0]),
        "augrc": 0,
    },
    SC_test(c_all_equal, p_all_zero): {
        "ID": "confid-all-equal_p-all-zero",
        "aurc": 1.0,
        "eaurc": 0.0,
        "aurc_achievable": 1.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 1.0, c_all_equal[0]),
        "selective-risk@95cov": (1.0, 1.0, c_all_equal[0]),
        "augrc": 0.5,
    },
    SC_test(c_all_equal, p_alternating_01): {
        "ID": "confid-all-equal_p-01",
        "aurc": 0.5,
        "eaurc": 0.3465735902799724,
        "aurc_achievable": 0.5,
        "eaurc_achievable": 0.3465735902799724,
        "selective-risk@50cov": (1.0, 0.5, c_all_equal[0]),
        "selective-risk@95cov": (1.0, 0.5, c_all_equal[0]),
        "augrc": 0.25,
    },
    SC_test(c_all_equal, p_some_nan): {
        "ID": "confid-all-equal_p-some-nan",
    },
    SC_test(c_all_equal, p_ascending): {
        "ID": "confid-all-equal_p-asc",
        "aurc": 0.5,
        "eaurc": 0.2524999999999986,
        "aurc_achievable": 0.5,
        "eaurc_achievable": 0.2524999999999986,
        "selective-risk@50cov": (1.0, 0.5, c_all_equal[0]),
        "selective-risk@95cov": (1.0, 0.5, c_all_equal[0]),
        "augrc": 0.24999999999999997,
    },
    SC_test(c_all_equal, p_plateaus): {
        "ID": "confid-all-equal_p-plateau",
        "aurc": 0.5,
        "eaurc": 0.1817914679883058,
        "aurc_achievable": 0.5,
        "eaurc_achievable": 0.1817914679883058,
        "selective-risk@50cov": (1.0, 0.5, c_all_equal[0]),
        "selective-risk@95cov": (1.0, 0.5, c_all_equal[0]),
        "augrc": 0.25000000000000017,
    },
    # # # -- Confidence scores alternating starting with 0 ---------------------------------
    SC_test(c_alternating_01, p_all_one): {
        "ID": "confid-01_p-all-one",
        "aurc": 0.0,
        "eaurc": 0.0,
        "aurc_achievable": 0.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 0.0, 0.0),
        "selective-risk@95cov": (1.0, 0.0, 0.0),
        "augrc": 0,
    },
    SC_test(c_alternating_01, p_all_zero): {
        "ID": "confid-01_p-all-zero",
        "aurc": 1.0,
        "eaurc": 0.0,
        "aurc_achievable": 1.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 1.0, 0.0),
        "selective-risk@95cov": (1.0, 1.0, 0.0),
        "augrc": 0.5,
    },
    SC_test(c_alternating_01, p_alternating_01): {
        "ID": "confid-01_p-01",
        "aurc": 0.125,
        # NOTE Using linear interpolation for computing the AURC can lead to negative
        # e-AURC values
        "eaurc": -0.02842640972002758,
        # NOTE Slightly higher than aurc due to non-linear interpolation for achievable
        "aurc_achievable": 0.15342640972002733,
        "eaurc_achievable": 0,
        "selective-risk@50cov": (0.5, 0.0, 1.0),
        "selective-risk@95cov": (1.0, 0.5, 0.0),
        "augrc": 0.125,
    },
    SC_test(c_alternating_01, p_some_nan): {
        "ID": "confid-01_p-some-nan",
    },
    SC_test(c_alternating_01, p_ascending): {
        "ID": "confid-01_p-asc",
        "aurc": 0.49621212121212127,
        "eaurc": 0.24871212121211994,
        # NOTE Slightly higher than aurc due to non-linear interpolation for achievable
        "aurc_achievable": 0.4964992566638387,
        "eaurc_achievable": 0.24899925666383738,
        "selective-risk@50cov": (0.5, 0.4949495, 1.0),
        "selective-risk@95cov": (1.0, 0.5, 0.0),
        "augrc": 0.24873737373737373,
    },
    SC_test(c_alternating_01, p_plateaus): {
        "ID": "confid-01_p-plateau",
        "aurc": 0.49699999999999944,
        "eaurc": 0.17879146798830492,
        "aurc_achievable": 0.4972274112777597,
        "eaurc_achievable": 0.1790188792660652,
        "selective-risk@50cov": (0.5, 0.496, 1.0),
        "selective-risk@95cov": (1.0, 0.5, 0.0),
        "augrc": 0.24899999999999978,
    },
    # -- Confidence scores alternating starting with 1 ---------------------------------
    SC_test(c_alternating_10, p_all_one): {
        "ID": "confid-10_p-all-one",
        "aurc": 0.0,
        "eaurc": 0.0,
        "aurc_achievable": 0.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 0.0, 0.0),
        "selective-risk@95cov": (1.0, 0.0, 0.0),
        "augrc": 0,
    },
    SC_test(c_alternating_10, p_all_zero): {
        "ID": "confid-10_p-all-zero",
        "aurc": 1.0,
        "eaurc": 0.0,
        "aurc_achievable": 1.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 1.0, 0.0),
        "selective-risk@95cov": (1.0, 1.0, 0.0),
        "augrc": 0.5,
    },
    SC_test(c_alternating_10, p_alternating_01): {
        "ID": "confid-10_p-01",
        "aurc": 0.875,
        "eaurc": 0.7215735902799725,
        "aurc_achievable": 0.5,
        "eaurc_achievable": 0.3465735902799724,
        "selective-risk@50cov": (1.0, 0.5, 0.0),
        "selective-risk@95cov": (1.0, 0.5, 0.0),
        "augrc": 0.375,
    },
    SC_test(c_alternating_10, p_some_nan): {
        "ID": "confid-10_p-some-nan",
    },
    SC_test(c_alternating_10, p_ascending): {
        "ID": "confid-10_p-asc",
        "aurc": 0.503787878787879,
        "eaurc": 0.2562878787878777,
        "aurc_achievable": 0.5035007433361615,
        "eaurc_achievable": 0.25600074333616013,
        "selective-risk@50cov": (1.0, 0.5, 0.0),
        "selective-risk@95cov": (1.0, 0.5, 0.0),
        "augrc": 0.25126262626262635,
    },
    SC_test(c_alternating_10, p_plateaus): {
        "ID": "confid-10_p-plateau",
        "aurc": 0.5029999999999994,
        "eaurc": 0.18479146798830492,
        "aurc_achievable": 0.5027725887222393,
        "eaurc_achievable": 0.18456405671054477,
        "selective-risk@50cov": (1.0, 0.5, 0.0),
        "selective-risk@95cov": (1.0, 0.5, 0.0),
        "augrc": 0.2509999999999998,
    },
    # -- Confidence scores ascending ---------------------------------------------------
    SC_test(c_ascending, p_all_one): {
        "ID": "confid-asc_p-all-one",
        "aurc": 0.0,
        "eaurc": 0.0,
        "aurc_achievable": 0.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 0.0, 0.0),
        "selective-risk@95cov": (1.0, 0.0, 0.0),
        "augrc": 0,
    },
    SC_test(c_ascending, p_all_zero): {
        "ID": "confid-asc_p-all-zero",
        "aurc": 1.0,
        "eaurc": 0.0,
        "aurc_achievable": 1.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 1.0, 0.0),
        "selective-risk@95cov": (1.0, 1.0, 0.0),
        "augrc": 0.5,
    },
    SC_test(c_ascending, p_alternating_01): {
        "ID": "confid-asc_p-01",
        "aurc": 0.48281112575762547,
        "eaurc": 0.3293847160375979,
        "aurc_achievable": 0.4719992328225763,
        "eaurc_achievable": 0.3185728231025487,
        "selective-risk@50cov": (0.51, 0.4901961, 0.4949495),
        "selective-risk@95cov": (0.95, 0.4947368, 0.0505051),
        "augrc": 0.2475,
    },
    SC_test(c_ascending, p_some_nan): {
        "ID": "confid-asc_p-some-nan",
    },
    SC_test(c_ascending, p_ascending): {
        "ID": "confid-asc_p-asc",
        "aurc": 0.24750000000000133,
        "eaurc": 0,
        "aurc_achievable": 0.24753863826243439,
        "eaurc_achievable": 3.8638262433055015e-05,
        "selective-risk@50cov": (0.5, 0.2474747, 0.5050505),
        "selective-risk@95cov": (0.95, 0.4747475, 0.0505051),
        "augrc": 0.16583333333333353,
    },
    SC_test(c_ascending, p_plateaus): {
        "ID": "confid-asc_p-plateau",
        "aurc": 0.3182085320116945,
        "eaurc": 0.0,
        "aurc_achievable": 0.31821825302024953,
        "eaurc_achievable": 9.721008555008126e-06,
        "selective-risk@50cov": (0.5, 0.3, 0.5050505),
        "selective-risk@95cov": (0.95, 0.4842105, 0.0505051),
        "augrc": 0.18750000000000108,
    },
    # -- Confidence scores ascending (unnormalized) ------------------------------------
    SC_test(c_ascending_unnormalized, p_all_one): {
        "ID": "confid-asc-u_p-all-one",
        "aurc": 0.0,
        "eaurc": 0.0,
        "aurc_achievable": 0.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 0.0, -50.0),
        "selective-risk@95cov": (1.0, 0.0, -50.0),
        "augrc": 0,
    },
    SC_test(c_ascending_unnormalized, p_all_zero): {
        "ID": "confid-asc-u_p-all-zero",
        "aurc": 1.0,
        "eaurc": 0.0,
        "aurc_achievable": 1.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 1.0, -50.0),
        "selective-risk@95cov": (1.0, 1.0, -50.0),
        "augrc": 0.5,
    },
    SC_test(c_ascending_unnormalized, p_alternating_01): {
        "ID": "confid-asc-u_p-01",
        "aurc": 0.48281112575762547,
        "eaurc": 0.3293847160375979,
        "aurc_achievable": 0.4719992328225763,
        "eaurc_achievable": 0.3185728231025487,
        "selective-risk@50cov": (0.51, 0.4901961, -1.0),
        "selective-risk@95cov": (0.95, 0.4947368, -45.0),
        "augrc": 0.2475,
    },
    SC_test(c_ascending_unnormalized, p_some_nan): {
        "ID": "confid-asc-u_p-some-nan",
    },
    SC_test(c_ascending_unnormalized, p_ascending): {
        "ID": "confid-asc-u_p-asc",
        "aurc": 0.24750000000000133,
        "eaurc": 0,
        "aurc_achievable": 0.24753863826243439,
        "eaurc_achievable": 3.8638262433055015e-05,
        "selective-risk@50cov": (0.5, 0.2474747, 0.0),
        "selective-risk@95cov": (0.95, 0.4747475, -45.0),
        "augrc": 0.16583333333333353,
    },
    SC_test(c_ascending_unnormalized, p_plateaus): {
        "ID": "confid-asc-u_p-plateau",
        "aurc": 0.3182085320116945,
        "eaurc": 0.0,
        "aurc_achievable": 0.31821825302024953,
        "eaurc_achievable": 9.721008555008126e-06,
        "selective-risk@50cov": (0.5, 0.3, 0.0),
        "selective-risk@95cov": (0.95, 0.4842105, -45.0),
        "augrc": 0.18750000000000108,
    },
    # -- Confidence scores some NaN ----------------------------------------------------
    SC_test(c_some_nan, p_all_one): {
        "ID": "confid-some-nan_p-all-one",
    },
    SC_test(c_some_nan, p_all_zero): {
        "ID": "confid-some-nan_p-all-zero",
    },
    SC_test(c_some_nan, p_alternating_01): {
        "ID": "confid-some-nan_p-01",
    },
    SC_test(c_some_nan, p_some_nan): {
        "ID": "confid-some-nan_p-some-nan",
    },
    SC_test(c_some_nan, p_ascending): {
        "ID": "confid-some-nan_p-asc",
    },
    SC_test(c_some_nan, p_plateaus): {
        "ID": "confid-some-nan_p-plateau",
    },
    # -- Confidence scores plateaus ----------------------------------------------------
    SC_test(c_plateaus, p_all_one): {
        "ID": "confid-plateaus_p-all-one",
        "aurc": 0.0,
        "eaurc": 0.0,
        "aurc_achievable": 0.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 0.0, 0.2),
        "selective-risk@95cov": (1.0, 0.0, 0.2),
        "augrc": 0,
    },
    SC_test(c_plateaus, p_all_zero): {
        "ID": "confid-plateaus_p-all-zero",
        "aurc": 1.0,
        "eaurc": 0.0,
        "aurc_achievable": 1.0,
        "eaurc_achievable": 0.0,
        "selective-risk@50cov": (1.0, 1.0, 0.2),
        "selective-risk@95cov": (1.0, 1.0, 0.2),
        "augrc": 0.5,
    },
    SC_test(c_plateaus, p_alternating_01): {
        "ID": "confid-plateaus_p-01",
        "aurc": 0.49083333333333334,
        "eaurc": 0.33740692361330576,
        "aurc_achievable": 0.4887532971076239,
        "eaurc_achievable": 0.3353268873875963,
        "selective-risk@50cov": (0.75, 0.4933333, 0.4),
        "selective-risk@95cov": (1.0, 0.5, 0.2),
        "augrc": 0.2475,
    },
    SC_test(c_plateaus, p_plateau_permutation): {
        "ID": "confid-plateaus_p-plateau-perm",
        "aurc": 0.49083333333333334,
        "eaurc": 0.33740692361330576,
        "aurc_achievable": 0.4887532971076239,
        "eaurc_achievable": 0.3353268873875963,
        "selective-risk@50cov": (0.75, 0.4933333, 0.4),
        "selective-risk@95cov": (1.0, 0.5, 0.2),
        "augrc": 0.2475,
    },
    SC_test(c_plateaus, p_some_nan): {
        "ID": "confid-plateaus_p-some-nan",
    },
    SC_test(c_plateaus, p_ascending): {
        "ID": "confid-plateaus_p-asc",
        "aurc": 0.2632575757575754,
        "eaurc": 0.015757575757574083,
        "aurc_achievable": 0.27047759219727724,
        "eaurc_achievable": 0.02297759219727591,
        "selective-risk@50cov": (0.5, 0.2474747, 0.6),
        "selective-risk@95cov": (1.0, 0.5, 0.2),
        "augrc": 0.17108585858585845,
    },
    SC_test(c_plateaus, p_plateaus): {
        "ID": "confid-plateaus_p-plateau",
        "aurc": 0.31250000000000255,
        "eaurc": -0.005708532011691969,
        "aurc_achievable": 0.31821825302024637,
        "eaurc_achievable": 9.72100855184399e-06,
        "selective-risk@50cov": (0.5, 0.3, 0.6),
        "selective-risk@95cov": (1.0, 0.5, 0.2),
        "augrc": 0.18750000000000097,
    },
}


# Testing metrics that explicitly depend on GT labels
RC_STATS_CLASS_AWARE_TEST_CASES = {}
