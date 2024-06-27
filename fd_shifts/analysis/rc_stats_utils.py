from typing import Any

import numpy as np
import numpy.typing as npt


def selective_risk_stats(
    *,
    confids: npt.NDArray[Any],
    residuals: npt.NDArray[Any],
    idx_sorted_confids: npt.NDArray[Any] = None,
) -> dict:
    """Computes the RC-point stats for the Selective Risk.

    Returns:
        dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
    """
    n = len(residuals)
    idx_sorted = (
        np.argsort(confids) if idx_sorted_confids is None else idx_sorted_confids
    )  # ascending scores
    residuals = residuals[idx_sorted]
    confidence = confids[idx_sorted]

    cov = n
    selective_risk_norm = n
    error_sum = sum(residuals)

    coverages = [1.0]  # descending coverage
    risks = [error_sum / cov]
    thresholds = [confidence[0]]
    working_point_mask = [True]
    current_min_risk = risks[0]

    for i in range(n - 1):
        cov -= 1
        selective_risk_norm -= 1
        error_sum -= residuals[i]

        if confidence[i] != confidence[i + 1]:
            selective_risk = error_sum / selective_risk_norm
            if selective_risk < current_min_risk:
                working_point_mask.append(True)
                current_min_risk = selective_risk
            else:
                working_point_mask.append(False)

            thresholds.append(confidence[i + 1])
            coverages.append(cov / n)
            risks.append(selective_risk)

    coverages.append(0)
    risks.append(risks[-1])
    working_point_mask.append(False)

    return {
        "coverages": np.array(coverages),
        "risks": np.array(risks),
        "thresholds": np.array(thresholds),
        "working_point_mask": working_point_mask,
    }


def generalized_risk_stats(
    *,
    confids: npt.NDArray[Any],
    residuals: npt.NDArray[Any],
    idx_sorted_confids: npt.NDArray[Any] = None,
) -> dict:
    """Computes the RC-point stats for the Generalized Risk.

    Returns:
        dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
    """
    n = len(residuals)
    idx_sorted = (
        np.argsort(confids) if idx_sorted_confids is None else idx_sorted_confids
    )  # ascending scores
    residuals = residuals[idx_sorted]
    confidence = confids[idx_sorted]

    cov = n
    error_sum = sum(residuals)

    coverages = [1.0]
    risks = [error_sum / n]
    thresholds = [confidence[0]]
    working_point_mask = [True]
    current_min_risk = risks[0]

    for i in range(n - 1):
        cov -= 1
        error_sum -= residuals[i]
        risk = error_sum / n
        if confidence[i] != confidence[i + 1]:
            if risk < current_min_risk:
                working_point_mask.append(True)
                current_min_risk = risk
            else:
                working_point_mask.append(False)

            thresholds.append(confidence[i + 1])
            coverages.append(cov / n)
            risks.append(risk)

    coverages.append(0)
    working_point_mask.append(risks[-1] > 0)
    risks.append(0)

    return {
        "coverages": np.array(coverages),
        "risks": np.array(risks),
        "thresholds": np.array(thresholds),
        "working_point_mask": working_point_mask,
    }


def selective_risk_ba_stats(
    *,
    confids: npt.NDArray[Any],
    residuals: npt.NDArray[Any],
    labels: npt.NDArray[Any] = None,
    idx_sorted_confids: npt.NDArray[Any] = None,
    drop_empty_classes: bool = False,
):
    """Computes the RC-point stats for the Selective Risk based on Balanced Accuracy.

    Returns:
        dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
    """
    assert labels is not None
    unique_labels, class_coverages = np.unique(labels, return_counts=True)
    # Set up look-up dict to access `class_coverages` based on labels
    label_to_idx = {l: idx for idx, l in enumerate(unique_labels)}

    n = len(confids)
    cov = n
    coverages = [1.0]

    idx_sorted = (
        np.argsort(confids) if idx_sorted_confids is None else idx_sorted_confids
    )  # ascending scores
    residuals = residuals[idx_sorted]
    confids = confids[idx_sorted]
    labels = labels[idx_sorted]

    error_sum = np.array([sum(residuals[labels == c]) for c in unique_labels])
    risks = [np.mean(error_sum / class_coverages)]
    thresholds = [confids[0]]
    working_point_mask = [True]
    current_min_risk = risks[0]

    for i in range(n - 1):
        c = label_to_idx[labels[i]]
        cov -= 1
        class_coverages[c] -= 1
        error_sum[c] -= residuals[i]

        # NOTE: If True, classes that are completely deferred no longer contribute to the
        #       score. But this would mean that CSFs are highly rewarded if they keep TPs
        #       from all classes up to high confidences.
        if drop_empty_classes and class_coverages[c] == 0:
            if not error_sum.dtype == "float":
                error_sum = np.array(error_sum, dtype=float)
            error_sum[c] = np.nan

        if confids[i] != confids[i + 1]:
            selective_recalls = np.divide(
                error_sum,
                class_coverages,
                out=np.zeros_like(error_sum, dtype=float),
                where=class_coverages != 0,
            )
            selective_risk = np.mean(selective_recalls)
            if selective_risk < current_min_risk:
                working_point_mask.append(True)
                current_min_risk = selective_risk
            else:
                working_point_mask.append(False)

            thresholds.append(confids[i + 1])
            coverages.append(cov / n)
            risks.append(selective_risk)

    coverages.append(0)
    risks.append(risks[-1])
    working_point_mask.append(False)

    return {
        "coverages": np.array(coverages),
        "risks": np.array(risks),
        "thresholds": np.array(thresholds),
        "working_point_mask": working_point_mask,
    }


def generalized_risk_ba_stats(
    *,
    confids: npt.NDArray[Any],
    residuals: npt.NDArray[Any],
    labels: npt.NDArray[Any] = None,
    idx_sorted_confids: npt.NDArray[Any] = None,
):
    """Computes the RC-point stats for the Generalized Risk based on Balanced Accuracy.

    Returns:
        dict with keys: "coverages", "risks", "thresholds", "working_point_mask"
    """
    assert labels is not None
    unique_labels, class_coverages = np.unique(labels, return_counts=True)
    # Set up look-up dict to access `class_coverages` based on labels
    label_to_idx = {l: idx for idx, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    n = len(confids)
    # Adjust residuals according to class prevalences: residuals for classes with
    # prevalence = 1/K stay the same.
    weights = np.array(
        [n / (num_classes * class_coverages[label_to_idx[l]]) for l in labels]
    )
    assert np.isclose(np.sum(weights), n)
    residuals = residuals * weights

    return generalized_risk_stats(
        confids=confids,
        residuals=residuals,
        idx_sorted_confids=idx_sorted_confids,
    )
