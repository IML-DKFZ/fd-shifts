import os
from pathlib import Path

from fd_shifts.configs import Config, DataConfig


def get_path(config: Config) -> Path | None:
    paths = os.getenv("FD_SHIFTS_STORE_PATH", "").split(":")
    for path in paths:
        path = Path(path)
        exp_path = path / config.exp.group_name / config.exp.name
        if (exp_path / "hydra" / "config.yaml").exists():
            return exp_path


def list_analysis_output_files(config: Config) -> list:
    files = []
    for study_name, testset in config.eval.query_studies:
        if study_name == "iid_study":
            files.append("analysis_metrics_iid_study.csv")
            continue
        if study_name == "noise_study":
            if isinstance(testset, DataConfig) and testset.dataset is not None:
                files.extend(
                    f"analysis_metrics_noise_study_{i}.csv" for i in range(1, 6)
                )
            continue

        if isinstance(testset, list):
            if len(testset) > 0:
                if isinstance(testset[0], DataConfig):
                    testset = map(
                        lambda d: d.dataset
                        + (
                            "_384"
                            if d.img_size[0] == 384 and "384" not in d.dataset
                            else ""
                        ),
                        testset,
                    )

                testset = [f"analysis_metrics_{study_name}_{d}.csv" for d in testset]
                if study_name == "new_class_study":
                    testset = [
                        d.replace(".csv", f"_{mode}.csv")
                        for d in testset
                        for mode in ["original_mode", "proposed_mode"]
                    ]
                files.extend(list(testset))
        elif isinstance(testset, DataConfig) and testset.dataset is not None:
            files.append(testset.dataset)
        elif isinstance(testset, str):
            files.append(testset)

    if config.eval.val_tuning:
        files.append("analysis_metrics_val_tuning.csv")

    return files


def list_bootstrap_analysis_output_files(
    config: Config,
    filter_study_name: list = None,
    original_new_class_mode: bool = False,
) -> list:
    subdir = "bootstrap/"
    files = []
    for study_name, testset in config.eval.query_studies:
        # Keep only studies that are in filter_study_name
        if filter_study_name is not None and study_name not in filter_study_name:
            continue

        if study_name == "iid_study":
            files.append(subdir + "analysis_metrics_iid_study.csv")
            continue
        if study_name == "noise_study":
            if isinstance(testset, DataConfig) and testset.dataset is not None:
                files.extend(
                    subdir + f"analysis_metrics_noise_study_{i}.csv"
                    for i in range(1, 6)
                )
            continue

        if isinstance(testset, list):
            if len(testset) > 0:
                if isinstance(testset[0], DataConfig):
                    testset = map(
                        lambda d: d.dataset + ("_384" if d.img_size[0] == 384 else ""),
                        testset,
                    )

                testset = [
                    subdir + f"analysis_metrics_{study_name}_{d}.csv" for d in testset
                ]
                if study_name == "new_class_study":
                    testset = [
                        d.replace(
                            ".csv",
                            (
                                "_original_mode.csv"
                                if original_new_class_mode
                                else "_proposed_mode.csv"
                            ),
                        )
                        for d in testset
                    ]
                files.extend(list(testset))
        elif isinstance(testset, DataConfig) and testset.dataset is not None:
            files.append(subdir + testset.dataset)
        elif isinstance(testset, str):
            files.append(subdir + testset)

    if config.eval.val_tuning:
        files.append(subdir + "analysis_metrics_val_tuning.csv")

    return files
