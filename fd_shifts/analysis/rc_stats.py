import logging
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull
from sklearn import metrics
from sklearn.utils import resample

from .rc_stats_utils import (
    generalized_risk_ba_stats,
    generalized_risk_stats,
    selective_risk_ba_stats,
    selective_risk_stats,
)


class RiskCoverageStatsMixin:
    """Mixin for statistics related to the Risk-Coverage-Curve. Classes that inherit from
    RiskCoverageStatsMixin should provide the following members:

    - ``residuals``, array of shape (N,): Residuals (binary or non-binary)
    - ``confids``, array of shape (N,): Confidence scores
    - ``labels``, array of shape (N,): Class labels (required for class-specific risks)
    """

    AUC_DISPLAY_SCALE: int = 1000
    RAISE_ON_NAN: bool = False

    def __init__(self):
        super().__init__()

    @cached_property
    def n(self) -> int:
        """Number of predictions"""
        return len(self.residuals)

    @cached_property
    def contains_nan(self) -> bool:
        """Whether the residuals or confidence scores contain NaN values"""
        return any(np.isnan(self.residuals)) or any(np.isnan(self.confids))

    @cached_property
    def is_binary(self) -> bool:
        """Whether the residuals are binary"""
        return np.all(np.logical_or(self.residuals == 0, self.residuals == 1))

    @property
    def _idx_sorted_confids(self) -> npt.NDArray[Any]:
        """Indices that sort the confidence scores in ascending order"""
        return np.argsort(self.confids)

    @property
    def _idx_sorted_residuals(self) -> npt.NDArray[Any]:
        """Indices that sort the residuals in ascending order"""
        return np.argsort(self.residuals)

    @cached_property
    def curve_stats_selective_risk(self) -> dict:
        """RC curve stats for selective risk.

        Returns:
            dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
        """
        self._validate()
        return self._evaluate_rc_curve_stats(risk="selective-risk")

    @cached_property
    def curve_stats_generalized_risk(self) -> dict:
        """RC curve stats for generalized risk.

        Returns:
            dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
        """
        self._validate()
        return self._evaluate_rc_curve_stats(risk="generalized-risk")

    @cached_property
    def curve_stats_selective_risk_ba(self) -> dict:
        """RC curve stats for selective risk with BA.

        Returns:
            dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
        """
        self._validate()
        return self._evaluate_rc_curve_stats(risk="selective-risk-ba")

    @cached_property
    def curve_stats_generalized_risk_ba(self) -> dict:
        """RC curve stats for generalized risk with BA.

        Returns:
            dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
        """
        self._validate()
        return self._evaluate_rc_curve_stats(risk="generalized-risk-ba")

    @property
    def coverages(self) -> npt.NDArray[Any]:
        """Coverage values in [0, 1], descending"""
        return self.curve_stats_generalized_risk["coverages"]

    @property
    def thresholds(self) -> npt.NDArray[Any]:
        """Confidence threshold values, ascending"""
        return self.curve_stats_generalized_risk["thresholds"]

    @property
    def working_point_mask(self) -> list[bool]:
        """Boolean array indicating the potential working points"""
        return self.curve_stats_generalized_risk["working_point_mask"]

    @property
    def selective_risks(self) -> npt.NDArray[Any]:
        """Selective risk values in [0, 1], sorted by ascending confidence"""
        return self.curve_stats_selective_risk["risks"]

    @property
    def generalized_risks(self) -> npt.NDArray[Any]:
        """Generalized risk values in [0, 1], sorted by ascending confidence"""
        return self.curve_stats_generalized_risk["risks"]

    @property
    def selective_risks_ba(self) -> npt.NDArray[Any]:
        """Selective BA-Risk values in [0, 1], sorted by ascending confidence"""
        return self.curve_stats_selective_risk_ba["risks"]

    @property
    def generalized_risks_ba(self) -> npt.NDArray[Any]:
        """Generalized BA-Risk values in [0, 1], sorted by ascending confidence"""
        return self.curve_stats_generalized_risk_ba["risks"]

    @cached_property
    def aurc(self) -> float:
        """Area under Risk Coverage Curve"""
        return self.evaluate_auc(risk="selective-risk")

    @cached_property
    def aurc_achievable(self) -> float:
        """Achievable area under Risk Coverage Curve"""
        return self.evaluate_auc(
            risk="selective-risk", achievable=True, interpolation="non-linear"
        )

    @cached_property
    def eaurc(self) -> float:
        """Excess AURC"""
        return self.aurc - self.aurc_optimal

    @cached_property
    def eaurc_achievable(self) -> float:
        """Achievable excess AURC"""
        return self.aurc_achievable - self.aurc_optimal

    @cached_property
    def augrc(self) -> float:
        """Area under Generalized Risk Coverage Curve"""
        return self.evaluate_auc(risk="generalized-risk")

    @cached_property
    def eaugrc(self) -> float:
        """Excess AUGRC"""
        return self.augrc - self.augrc_optimal

    @cached_property
    def aurc_ba(self) -> float:
        """AURC with Selective Balanced Accuracy"""
        return self.evaluate_auc(risk="selective-risk-ba")

    @cached_property
    def augrc_ba(self) -> float:
        """AUGRC with residuals corresponding to the Balanced Accuracy-residuals"""
        return self.evaluate_auc(risk="generalized-risk-ba")

    @cached_property
    def aurc_ci_bs(self) -> Tuple[float, float]:
        """Bootstrapped CI (95% percentiles) of the AURC"""
        return self.evaluate_ci(risk="selective-risk")

    @cached_property
    def augrc_ci_bs(self) -> Tuple[float, float]:
        """Bootstrapped CI (95% percentiles) of the AUGRC"""
        return self.evaluate_ci(risk="generalized-risk")

    @cached_property
    def aurc_ba_ci_bs(self) -> Tuple[float, float]:
        """Bootstrapped CI (95% percentiles) of the AURC-BA"""
        return self.evaluate_ci(risk="selective-risk-ba")

    @cached_property
    def augrc_ba_ci_bs(self) -> Tuple[float, float]:
        """Bootstrapped CI (95% percentiles) of the AUGRC-BA"""
        return self.evaluate_ci(risk="generalized-risk-ba")

    @cached_property
    def dominant_point_mask(self) -> list[bool]:
        """Boolean array masking the dominant RC-points"""
        if self.is_binary and not self.contains_nan:
            num_rc_points = len(self.coverages)

            if sum(self.residuals) in (0, self.n):
                # If the predictions are all correct or all wrong, the RC-Curve is a
                # horizontal line, and thus there is one dominant point at cov=1.
                indices = np.array([-1])
            else:
                # Compute the convex hull in ROC-space, as the dominant points are the
                # same in RC-space. Inspired by
                # https://github.com/foxtrotmike/rocch/blob/master/rocch.py
                fpr, tpr, _ = metrics.roc_curve(
                    1 - self.residuals, self.confids, drop_intermediate=False
                )
                if num_rc_points == 2:
                    # If there is only one point, the convex hull is trivial
                    return np.array([True, False])
                else:
                    # Add the (2, -1) point to make the convex hull construction easier.
                    fpr = np.concatenate((fpr, [2.0]))
                    tpr = np.concatenate((tpr, [-1.0]))
                    hull = ConvexHull(
                        np.concatenate((fpr.reshape(-1, 1), tpr.reshape(-1, 1)), axis=1)
                    )
                    indices = hull.vertices
                    indices = indices[(indices != 0) & (indices != num_rc_points)]

            mask = np.zeros(num_rc_points, dtype=bool)
            mask[indices] = True
            # Reverse the order (corresponding to descending coverage)
            return mask[::-1]

        # NOTE: For non-binary residuals, finding the subset of RC-points that minimizes
        #       the AURC is not straightforward.
        #       Don't mask any points in this case (only cov=0).
        mask = np.ones(len(self.coverages), dtype=bool)
        mask[-1] = 0
        return mask

    @cached_property
    def aurc_optimal(self) -> float:
        """AURC for the same prediction values but optimal confidence scores. Used as
        reference for e-AURC calculation.

        For binary residuals, the analytical formula (based on accuracy) is used.
        Otherwise, the optimal AURC is calculated based on ideally sorted and stratified
        scores.

        Note that if there are confidence plateaus, the computed optimal AURC may be
        higher, yielding a negative e-AURC.
        """
        if self.contains_nan:
            return np.nan

        if self.is_binary:
            # Directly calculate optimal AURC from accuracy
            err = np.mean(self.residuals)
            return self.AUC_DISPLAY_SCALE * (
                err + (1 - err) * (np.log(1 - err + np.finfo(err.dtype).eps))
            )

        # Evaluate the AURC for optimal confidence scores
        rc_point_stats_optimal = self._evaluate_rc_curve_stats(
            risk="selective-risk",
            confids=np.linspace(1, 0, len(self.confids)),
            residuals=self.residuals[self._idx_sorted_residuals],
            labels=(
                self.labels[self._idx_sorted_residuals]
                if self.labels is not None
                else None
            ),
        )
        return self.evaluate_auc(
            coverages=rc_point_stats_optimal["coverages"],
            risks=rc_point_stats_optimal["risks"],
        )

    @cached_property
    def augrc_optimal(self) -> float:
        """AUGRC for the same prediction values but optimal confidence scores. Used as
        reference for e-AUGRC calculation.
        """
        if self.contains_nan:
            return np.nan

        if self.is_binary:
            return 0.5 * np.mean(self.residuals) ** 2 * self.AUC_DISPLAY_SCALE

        rc_point_stats_optimal = self._evaluate_rc_curve_stats(
            risk="generalized-risk",
            confids=np.linspace(1, 0, len(self.confids)),
            residuals=self.residuals[self._idx_sorted_residuals],
            labels=(
                self.labels[self._idx_sorted_residuals]
                if self.labels is not None
                else None
            ),
        )
        return self.evaluate_auc(
            coverages=rc_point_stats_optimal["coverages"],
            risks=rc_point_stats_optimal["risks"],
        )

    def _validate(self) -> None:
        """"""
        assert hasattr(self, "residuals"), "Missing class member 'residuals'"
        assert hasattr(self, "confids"), "Missing class member 'confids'"

        if self.contains_nan:
            msg = (
                f"There are {sum(np.isnan(self.confids))} NaN confidence values and "
                f"{sum(np.isnan(self.residuals))} NaN residuals."
            )
            if self.RAISE_ON_NAN:
                raise ValueError(msg)
            else:
                logging.warning(msg)

    def _evaluate_rc_curve_stats(
        self,
        *,
        risk: str,
        confids: npt.NDArray[Any] = None,
        residuals: npt.NDArray[Any] = None,
        labels: npt.NDArray[Any] = None,
    ) -> dict:
        """Computes the RC-points and the corresponding thresholds and working point mask.

        Returns:
            dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
        """
        logging.debug("Evaluating the RC points ...")

        idx_sorted_confids = (
            np.argsort(confids) if confids is not None else self._idx_sorted_confids
        )
        confids = confids if confids is not None else self.confids
        residuals = residuals if residuals is not None else self.residuals
        labels = labels if labels is not None else self.labels

        if risk == "selective-risk":
            return selective_risk_stats(
                confids=confids,
                residuals=residuals,
                idx_sorted_confids=idx_sorted_confids,
            )
        elif risk == "generalized-risk":
            return generalized_risk_stats(
                confids=confids,
                residuals=residuals,
                idx_sorted_confids=idx_sorted_confids,
            )
        elif risk == "selective-risk-ba":
            return selective_risk_ba_stats(
                confids=confids,
                residuals=residuals,
                labels=labels,
                idx_sorted_confids=idx_sorted_confids,
            )
        elif risk == "generalized-risk-ba":
            return generalized_risk_ba_stats(
                confids=confids,
                residuals=residuals,
                labels=labels,
                idx_sorted_confids=idx_sorted_confids,
            )
        else:
            raise ValueError(f"Unknown risk type '{risk}'")

    def get_curve_stats(self, *, risk: str):
        """"""
        if risk == "selective-risk":
            return self.curve_stats_selective_risk
        elif risk == "generalized-risk":
            return self.curve_stats_generalized_risk
        elif risk == "selective-risk-ba":
            return self.curve_stats_selective_risk_ba
        elif risk == "generalized-risk-ba":
            return self.curve_stats_generalized_risk_ba
        else:
            raise ValueError(f"Unknown risk type '{risk}'")

    def evaluate_auc(
        self,
        *,
        risk: str = None,
        coverages: npt.NDArray[Any] = None,
        risks: npt.NDArray[Any] = None,
        cov_min=0,
        cov_max=1,
        achievable=False,
        interpolation: str = "linear",
    ) -> float:
        """Compute an AUC value. By default, it is computed over the whole coverage range
        [0, 1].

        Args:
            risk (str): Risk type (e.g. "selective-risk" for AURC, "generalized-risk" for
                AUGRC)
            coverages (npt.NDArray[Any], optional): coverage values
            risks (npt.NDArray[Any], optional): risk values
            cov_min (int, optional): Lower coverage limit. Defaults to 0.
            cov_max (int, optional): Upper coverage limit. Defaults to 1.
            achievable (bool, optional): Whether to compute the achievable AURC.
                Defaults to False.
            interpolation (str): Defaults to trapezoidal interpolation of the RC curve.

        Returns:
            float: Area under Risk Coverage Curve
        """
        if self.contains_nan:
            return np.nan

        if cov_max <= cov_min or cov_max <= 0 or cov_min >= 1:
            return 0.0

        assert (coverages is None) == (risks is None)
        if coverages is None:
            curve_stats = self.get_curve_stats(risk=risk)
            coverages = curve_stats["coverages"]
            risks = curve_stats["risks"]

        if achievable:
            if interpolation == "linear" and "generalized" not in risk:
                logging.warning(
                    "Achievable AURC values should be estimated with 'non-linear' "
                    f"interpolation. Currvently using: '{interpolation}' interpolation"
                )

            risks = risks[self.dominant_point_mask]
            coverages = coverages[self.dominant_point_mask]

        if interpolation == "linear" or "generalized" in risk:
            # Linear interpolation
            if cov_max != 1 or cov_min != 0:
                raise NotImplementedError()
            return -np.trapz(risks, coverages) * self.AUC_DISPLAY_SCALE

        # Removing the cov=0 point, this curve segment is handled separately
        if coverages[-1] == 0:
            coverages = coverages[:-1]
            risks = risks[:-1]

        # Non-linear interpolation for selective-risk-based AUC
        # Prepare the AURC evaluation for a certain coverage range
        n = self.n
        cov_below = 0
        error_sum_below = 0
        lower_lim = 0
        cov_above = None
        error_sum_above = 1
        upper_lim = 1

        if cov_min > 0:
            idx_range = np.argwhere(coverages >= cov_min)[:, 0]
            if idx_range[-1] < len(coverages) - 1:
                cov_below = coverages[idx_range[-1] + 1]
                error_sum_below = risks[idx_range[-1] + 1] * cov_below
            lower_lim = (cov_min - cov_below) / (coverages[idx_range[-1]] - cov_below)
            cov_below *= n
            error_sum_below *= n
            coverages = coverages[idx_range]
            risks = risks[idx_range]

        if cov_max < 1:
            idx_range = np.argwhere(coverages <= cov_max)[:, 0]
            if len(idx_range) > 0:
                cov_above = coverages[idx_range[0] - 1]
                error_sum_above = risks[idx_range[0] - 1] * cov_above
                upper_lim = (cov_max - coverages[idx_range[0]]) / (
                    cov_above - coverages[idx_range[0]]
                )
                cov_above *= n
                error_sum_above *= n
            else:
                cov_above = coverages[-1] * n
                error_sum_above = risks[-1] * cov_above
                upper_lim = cov_max / coverages[-1]

            coverages = coverages[idx_range]
            risks = risks[idx_range]

        # Integrate segments between RC-points
        cov = coverages * n
        error_sum = risks * cov

        # If cov is empty, integrate withing a single segment
        if len(cov) == 0:
            if cov_below == 0:
                aurc = error_sum_above * (upper_lim - lower_lim)
                return aurc / n
            else:
                cov_diff = cov_above - cov_below
                aurc = (error_sum_above - error_sum_below) * (upper_lim - lower_lim) + (
                    error_sum_below * cov_above - error_sum_above * cov_below
                ) * np.log(
                    (cov_above + cov_diff * upper_lim)
                    / (cov_below + cov_diff * lower_lim)
                ) / cov_diff
                return aurc / n

        aurc = 0
        # Add contributions of complete segments
        if len(cov) > 1:
            cov_diff = -np.diff(cov)
            error_sum_prev = error_sum[:-1]
            error_sum_next = error_sum[1:]
            cov_prev = cov[:-1]
            cov_next = cov[1:]

            aurc += sum(
                error_sum_prev
                - error_sum_next
                + (error_sum_next * cov_prev - error_sum_prev * cov_next)
                * np.log(cov_prev / cov_next)
                / cov_diff
            )

        # Additional contributions at lower coverage
        if cov_below == 0:
            aurc += error_sum[-1] * (1 - lower_lim)
        else:
            cov_diff = cov[-1] - cov_below
            aurc += (error_sum[-1] - error_sum_below) * (1 - lower_lim) + (
                error_sum_below * cov[-1] - error_sum[-1] * cov_below
            ) * np.log(cov[-1] / (cov_below + cov_diff * lower_lim)) / cov_diff

        # Additional contributions at higher coverage
        if cov_max < 1:
            cov_diff = cov_above - cov[0]
            aurc += (error_sum_above - error_sum[0]) * upper_lim + (
                error_sum[0] * cov_above - error_sum_above * cov[0]
            ) * np.log((cov[0] + cov_diff * upper_lim) / cov[0]) / cov_diff

        return aurc / n * self.AUC_DISPLAY_SCALE

    def evaluate_ci(
        self,
        *,
        risk: str,
        confids: npt.NDArray[Any] = None,
        residuals: npt.NDArray[Any] = None,
        labels: npt.NDArray[Any] = None,
        n_bs: int = 10000,
        stratified: bool = False,
    ):
        """Compute confidence intervals based on bootstrapping."""
        confids = confids if confids is not None else self.confids
        residuals = residuals if residuals is not None else self.residuals
        labels = labels if labels is not None else self.labels
        N = len(confids)
        aurc_bs = np.empty(n_bs)
        for i in range(n_bs):
            if not stratified or not self.is_binary:
                indices_bs = np.random.choice(np.arange(N), size=N, replace=True)
            else:
                indices_bs = resample(np.arange(N), n_samples=N, stratify=residuals)
            confids_bs = confids[indices_bs]
            residuals_bs = residuals[indices_bs]
            labels_bs = labels[indices_bs] if labels is not None else None

            curve_stats = self._evaluate_rc_curve_stats(
                risk=risk,
                confids=confids_bs,
                residuals=residuals_bs,
                labels=labels_bs,
            )
            aurc_bs[i] = self.evaluate_auc(
                coverages=curve_stats["coverages"], risks=curve_stats["risks"]
            )

        # Compute the empirical 95% quantiles
        return np.percentile(aurc_bs, [2.5, 97.5])

    def get_working_point(
        self,
        *,
        risk: str,
        target_risk=None,
        target_cov=None,
    ) -> tuple[float, float, float]:
        """Select a working point from the RC-points given a desired risk or coverage.

        Args:
            risk str: Risk type (e.g. "selective-risk" for AURC, "generalized-risk" for
                AUGRC)
            target_risk (float, optional): Desired (maximum) risk value in range [0, 1]
            target_cov (float, optional): Desired (maximum) coverage value in range [0, 1]

        Returns:
            working point (tuple): coverage, risk, threshold
        """
        if target_risk is None and target_cov is None:
            raise ValueError("Must provide either target_risk or target_cov value")
        if target_risk is not None and target_cov is not None:
            raise ValueError(
                "The target_risk and target_cov arguments are mutually exclusive"
            )

        curve_stats = self.get_curve_stats(risk=risk)
        working_point_mask = curve_stats["working_point_mask"]
        coverages = curve_stats["coverages"][working_point_mask]
        risks = curve_stats["risks"][working_point_mask]
        thresholds = np.r_[curve_stats["thresholds"], -np.infty][working_point_mask]

        if self.contains_nan:
            return np.nan, np.nan, np.nan

        if target_risk is not None:
            mask = np.argwhere(risks <= target_risk)[:, 0]
            idx = np.argmax(coverages[mask])
        elif target_cov is not None:
            mask = np.argwhere(coverages >= target_cov)[:, 0]
            idx = np.argmin(risks[mask])

        cov_value = coverages[mask][idx]
        risk_value = risks[mask][idx]
        threshold = thresholds[mask][idx]

        return cov_value, risk_value, threshold


class RiskCoverageStats(RiskCoverageStatsMixin):
    """Standalone RiskCoverageStats class"""

    def __init__(
        self,
        confids: npt.NDArray[Any],
        residuals: npt.NDArray[Any],
        labels: npt.NDArray[Any] = None,
    ):
        """Returns a RiskCoverageStats instance which allows for calculating metrics
        related to the Risk-Coverage-Curve.

        Applicable to binary failure labels as well as continuous residuals.

        Args:
            confids (npt.NDArray[Any]): Confidence values
            residuals (npt.NDArray[Any]): 'Residual scores' in [0, 1]. E.g., an integer
                array indicating wrong predictions. The selective risks are calculated
                as the sum of residuals after selection divided by the coverage.
            labels (npt.NDArray[Any], optional): Class labels (required for BA-based
                metrics and prevalence shifts)
        """
        super().__init__()
        self.confids = confids
        self.residuals = residuals
        self.labels = labels
