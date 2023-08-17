import math

import numpy as np
import pytest

from fd_shifts.analysis.metrics import StatsCache, aurc, baurc

"""
Tests to run

Inputs: 
    Labels: Int array, indicate class of sample
    Correct: binary int array, indicates if sample was correctly classified or not
    Confids: float array [0,1],Confidence in the sample
Outputs:
    B-AURC
    AURC
Test Cases:
    If correct = 1, B-AURC == 0
    If correct = 0, B-AURC == 1000 (or value close due to calculation of Area under Curve)
    If per class accuracies (p.c.a) equal and random confidence (r.c.) of class, and class equal prevalence: B-AURC ~ AURC
    If p.c.a. equal and r.c.  and class imbalance: B-AURC ~ AURC
    If p.c.a. better for low prev class: B~AURC < AURC
    If p.c.a. worse for low prev class: B~AURC > AURC
    If p.c.a. equal, but low prev class in beginning (e.g. lower confidence): B~AURC(front) < B-AURC(random)
"""


@pytest.mark.baurc
@pytest.mark.parametrize("correct", [np.ones(1000)])
def test_all_correct(correct):
    confids = np.random.uniform(size=1000)
    labels = np.random.randint(2, size=1000)
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    aurc_value = aurc(stat)
    baurc_value = baurc(stat)

    assert aurc_value == 0
    assert baurc_value == 0


@pytest.mark.baurc
@pytest.mark.parametrize("correct", [np.zeros(1000)])
def test_all_false(correct):
    confids = np.random.uniform(size=1000)
    labels = np.random.randint(2, size=1000)
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    aurc_value = aurc(stat)
    baurc_value = baurc(stat)

    assert aurc_value == pytest.approx(999, 1)
    assert baurc_value == pytest.approx(999, 1)
    assert baurc_value == aurc_value


@pytest.mark.baurc
@pytest.mark.parametrize("correct", [np.random.randint(2, size=1000)])
def test_all_random(correct):
    confids = np.random.uniform(size=1000)
    labels = np.random.randint(2, size=1000)
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    aurc_value = aurc(stat)
    baurc_value = baurc(stat)

    assert baurc_value == pytest.approx(aurc_value, 4)


@pytest.mark.baurc
@pytest.mark.parametrize("correct", [np.random.randint(2, size=1000)])
def test_all_class_imbalance_equal_acc(correct):
    confids = np.random.uniform(size=1000)
    labels = np.random.choice(2, size=1000, replace=True, p=[0.9, 0.1])
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    aurc_value = aurc(stat)
    baurc_value = baurc(stat)

    assert baurc_value == pytest.approx(aurc_value, 4)


@pytest.mark.baurc
def test_all_class_imbalance__low_higher_equal_acc():
    confids = np.random.uniform(size=1000)
    labels = np.random.choice(2, size=1000, replace=True, p=[0.9, 0.1])
    correct = np.zeros(1000)
    cor_cla_0 = np.random.choice(
        2, size=np.sum(labels == 0), replace=True, p=[0.5, 0.5]
    )
    cor_cla_1 = np.random.choice(
        2, size=np.sum(labels == 1), replace=True, p=[0.1, 0.9]
    )
    correct[labels == 0] = cor_cla_0
    correct[labels == 1] = cor_cla_1
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    aurc_value = aurc(stat)
    baurc_value = baurc(stat)

    assert baurc_value < aurc_value


@pytest.mark.baurc
def test_all_class_imbalance__low_lowerer_equal_acc():
    confids = np.random.uniform(size=1000)
    labels = np.random.choice(2, size=1000, replace=True, p=[0.9, 0.1])
    correct = np.zeros(1000)
    cor_cla_0 = np.random.choice(
        2, size=np.sum(labels == 0), replace=True, p=[0.5, 0.5]
    )
    cor_cla_1 = np.random.choice(
        2, size=np.sum(labels == 1), replace=True, p=[0.9, 0.1]
    )
    correct[labels == 0] = cor_cla_0
    correct[labels == 1] = cor_cla_1
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    aurc_value = aurc(stat)
    baurc_value = baurc(stat)

    assert baurc_value > aurc_value


@pytest.mark.baurc
def test_lowprev_front_order_class_imbalance_low_lowerer_acc():
    confids = np.zeros(10000)
    ones = np.ones(1000)
    zeros = np.zeros(9000)
    labels = np.concatenate([ones, zeros])
    correct = np.zeros(10000)
    cor_cla_0 = np.random.choice(
        2, size=np.sum(labels == 0), replace=True, p=[0.3, 0.7]
    )
    cor_cla_1 = np.random.choice(
        2, size=np.sum(labels == 1), replace=True, p=[0.6, 0.4]
    )
    confids_cla_0 = np.random.uniform(0.5, 1, size=np.sum(labels == 0))
    confids_cla_1 = np.random.uniform(0, 0.5, size=np.sum(labels == 1))
    correct[labels == 0] = cor_cla_0
    correct[labels == 1] = cor_cla_1
    confids[labels == 0] = confids_cla_0
    confids[labels == 1] = confids_cla_1
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    baurc_value_front = baurc(stat)
    ##################################
    confids = np.random.uniform(0, 1, size=10000)
    ones = np.ones(1000)
    zeros = np.zeros(9000)
    labels = np.concatenate([ones, zeros])
    correct = np.zeros(10000)
    cor_cla_0 = np.random.choice(
        2, size=np.sum(labels == 0), replace=True, p=[0.3, 0.7]
    )
    cor_cla_1 = np.random.choice(
        2, size=np.sum(labels == 1), replace=True, p=[0.6, 0.4]
    )
    confids_cla_0 = np.random.uniform(0, 0.5, size=np.sum(labels == 0))
    confids_cla_1 = np.random.uniform(0.5, 1, size=np.sum(labels == 1))
    correct[labels == 0] = cor_cla_0
    correct[labels == 1] = cor_cla_1
    stat = StatsCache(confids=confids, correct=correct, n_bins=10, labels=labels)
    baurc_value_rand = baurc(stat)

    print(baurc_value_front, baurc_value_rand)
    assert baurc_value_front < baurc_value_rand
