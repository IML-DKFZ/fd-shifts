import os
from pathlib import Path
from typing import cast

import pandas as pd

from fd_shifts.experiments import Experiment, get_all_experiments
from fd_shifts.reporting import tables
from fd_shifts.reporting.plots import plot_rank_style, vit_v_cnn_box
from fd_shifts.reporting.tables import (
    paper_results,
    rank_comparison_metric,
    rank_comparison_mode,
)

DATASETS = (
    "svhn",
    "cifar10",
    "cifar100",
    "super_cifar100",
    "camelyon",
    "animals",
    "breeds",
)


def _filter_experiment_by_dataset(experiments: list[Experiment], dataset: str):
    match dataset:
        case "super_cifar100":
            _experiments = list(
                filter(
                    lambda exp: exp.dataset in ("super_cifar100", "supercifar"),
                    experiments,
                )
            )
        case "animals":
            _experiments = list(
                filter(
                    lambda exp: exp.dataset in ("animals", "wilds_animals"), experiments
                )
            )
        case "animals_openset":
            _experiments = list(
                filter(
                    lambda exp: exp.dataset
                    in ("animals_openset", "wilds_animals_openset"),
                    experiments,
                )
            )
        case "camelyon":
            _experiments = list(
                filter(
                    lambda exp: exp.dataset in ("camelyon", "wilds_camelyon"),
                    experiments,
                )
            )
        case _:
            _experiments = list(filter(lambda exp: exp.dataset == dataset, experiments))

    return _experiments


def gather_data(data_dir: Path):
    """Collect all csv files from experiments into one location

    Args:
        data_dir (Path): where to collect to
    """
    experiment_dir = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    experiments = get_all_experiments()

    for dataset in DATASETS + ("animals_openset", "svhn_openset"):
        print(dataset)
        _experiments = _filter_experiment_by_dataset(experiments, dataset)

        _paths = []
        _vit_paths = []

        for experiment in _experiments:
            if experiment.model == "vit":
                _vit_paths.extend(
                    (experiment_dir / experiment.to_path() / "test_results").glob(
                        "*.csv"
                    )
                )
            else:
                _paths.extend(
                    (experiment_dir / experiment.to_path() / "test_results").glob(
                        "*.csv"
                    )
                )

        if len(_paths) > 0:
            dframe: pd.DataFrame = pd.concat(
                [cast(pd.DataFrame, pd.read_csv(p)) for p in _paths]
            )
            dframe.to_csv(data_dir / f"{dataset}.csv")

        if len(_vit_paths) > 0:
            dframe: pd.DataFrame = pd.concat(
                [cast(pd.DataFrame, pd.read_csv(p)) for p in _vit_paths]
            )
            dframe.to_csv(data_dir / f"{dataset}vit.csv")


def load_file(path: Path, experiment_override: str | None = None) -> pd.DataFrame:
    """Load experiment result csv into dataframe and set experiment accordingly

    Args:
        path (Path): path to csv file
        experiment_override (str | None): use this experiment instead of inferring it from the file

    Returns:
        Dataframe created from csv including some cleanup

    Raises:
        FileNotFoundError: if the file at path does not exist
        RuntimeError: if loading does not result in a dataframe
    """
    result = pd.read_csv(path)

    if not isinstance(result, pd.DataFrame):
        raise FileNotFoundError

    result = (
        result.assign(
            experiment=experiment_override
            if experiment_override is not None
            else path.stem
        )
        .dropna(subset=["name", "model"])
        .drop_duplicates(subset=["name", "study", "model", "network", "confid"])
    )

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError

    return result


def load_data(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Args:
        data_dir (Path): the directory where all experiment results are

    Returns:
        dataframe with all experiments and list of experiments that were loaded

    """
    data = pd.concat(
        [
            load_file(path)
            for path in filter(
                lambda path: str(path.stem).startswith(DATASETS),
                data_dir.glob("*.csv"),
            )
        ]
    )

    data = data.loc[~data["study"].str.contains("tinyimagenet_original")]
    data = data.loc[~data["study"].str.contains("tinyimagenet_proposed")]

    data = data.query(
        'not (experiment in ["cifar10", "cifar100", "super_cifar100"]'
        'and not name.str.contains("vgg13"))'
    )

    data = data.query(
        'not ((experiment.str.contains("super_cifar100")'
        ")"
        'and not (study == "iid_study"))'
    )

    data = data.query(
        'not (experiment.str.contains("openset")' 'and study.str.contains("iid_study"))'
    )

    data = data.assign(study=data.experiment + "_" + data.study)

    data = data.assign(
        study=data.study.mask(
            data.experiment == "super_cifar100",
            "cifar100_in_class_study_superclasses",
        ),
        experiment=data.experiment.mask(
            data.experiment == "super_cifar100", "cifar100"
        ),
    )

    data = data.assign(
        study=data.study.mask(
            data.experiment == "super_cifar100vit",
            "cifar100vit_in_class_study_superclasses",
        ),
        experiment=data.experiment.mask(
            data.experiment == "super_cifar100vit", "cifar100vit"
        ),
    )

    data = data.assign(ece=data.ece.mask(data.ece < 0))

    exp_names = list(
        filter(
            lambda exp: not exp.startswith("super_cifar100"),
            data.experiment.unique(),
        )
    )

    return data, exp_names


def _extract_hparam(
    name: pd.Series, regex: str, default: str | None = None
) -> pd.Series:
    result: pd.Series = name.str.replace(".*" + regex + ".*", "\\1", regex=True)
    return result


def assign_hparams_from_names(data: pd.DataFrame) -> pd.DataFrame:
    """Create columns for hyperparameters from experiment names

    Args:
        data (pd.DataFrame): experiment data

    Returns:
        experiment data with additional columns
    """
    data = data.assign(
        backbone=lambda data: _extract_hparam(
            data.name, r"bb([a-z0-9]+)(_small_conv)?"
        ),
        # Prefix model name with vit_ if it is a vit model
        # If it isn't a vit model, model is the first part of the name
        model=lambda data: data["backbone"]
        .mask(data["backbone"] != "vit", "")
        .mask(data["backbone"] == "vit", "vit_")
        + data.model.where(
            data.backbone == "vit", data.name.str.split("_", expand=True)[0]
        ),
        run=lambda data: _extract_hparam(data.name, r"run([0-9]+)"),
        dropout=lambda data: _extract_hparam(data.name, r"do([01])"),
        rew=lambda data: _extract_hparam(data.name, r"rew([0-9.]+)"),
        lr=lambda data: _extract_hparam(data.name, r"lr([0-9.]+)", "0.1"),
        # Encode every detail into confid name
        _confid=data.confid,
        confid=lambda data: data.model
        + "_"
        + data.confid
        + "_"
        + data.dropout
        + "_"
        + data.rew,
    )

    return data


def filter_best_lr(data: pd.DataFrame, metric: str = "aurc") -> pd.DataFrame:
    """
    for every study (which encodes dataset) and confidence (which encodes other stuff)
    select all runs with the best avg combo of reward and dropout

    Args:
        data (pd.DataFrame): experiment data
        metric (str): metric to select best from

    Returns:
        filtered data
    """

    def _filter_row(row, selection_df, optimization_columns, fixed_columns):
        if "openset" in row["study"]:
            return True
        if "superclasses" in row["study"]:
            return True
        if "vit" not in row["model"]:
            return True
        temp = selection_df[
            (row.experiment == selection_df.experiment)
            & (row._confid == selection_df._confid)
            & (row.model == selection_df.model)
        ]

        result = row[optimization_columns] == temp[optimization_columns]
        if result.all(axis=1).any().item():
            return True

        return False

    fixed_columns = [
        "study",
        "experiment",
        "_confid",
        "model",
        "rew",
        "dropout",
    ]
    optimization_columns = ["lr"]
    aggregation_columns = ["run", metric]

    # Only look at validation data and the relevant columns
    selection_df = data[data.study.str.contains("val_tuning")][
        fixed_columns + optimization_columns + aggregation_columns
    ]

    # compute aggregation column means
    selection_df = (
        selection_df.groupby(fixed_columns + optimization_columns).mean().reset_index()
    )

    # select best optimization columns combo
    selection_df = selection_df.iloc[
        selection_df.groupby(fixed_columns)[metric].idxmin()
    ]

    data = data[
        data.apply(
            lambda row: _filter_row(
                row, selection_df, optimization_columns, fixed_columns
            ),
            axis=1,
        )
    ]

    return data


def filter_best_hparams(data: pd.DataFrame, metric: str = "aurc") -> pd.DataFrame:
    """
    for every study (which encodes dataset) and confidence (which encodes other stuff)
    select all runs with the best avg combo of reward and dropout

    Args:
        data (pd.DataFrame): experiment data
        metric (str): metric to select best from

    Returns:
        filtered data
    """

    def _filter_row(row, selection_df, optimization_columns, fixed_columns):
        if "openset" in row["study"]:
            return True
        temp = selection_df[
            (row.experiment == selection_df.experiment)
            & (row._confid == selection_df._confid)
            & (row.model == selection_df.model)
        ]

        result = row[optimization_columns] == temp[optimization_columns]
        if result.all(axis=1).any().item():
            return True

        return False

    fixed_columns = [
        "study",
        "experiment",
        "_confid",
        "model",
    ]
    optimization_columns = ["rew", "dropout"]
    aggregation_columns = ["run", metric]

    # Only look at validation data and the relevant columns
    selection_df = data[data.study.str.contains("val_tuning")][
        fixed_columns + optimization_columns + aggregation_columns
    ]

    # compute aggregation column means
    selection_df = (
        selection_df.groupby(fixed_columns + optimization_columns).mean().reset_index()
    )

    # select best optimization columns combo
    selection_df = selection_df.iloc[
        selection_df.groupby(fixed_columns)[metric].idxmin()
    ]

    data = data[
        data.apply(
            lambda row: _filter_row(
                row, selection_df, optimization_columns, fixed_columns
            ),
            axis=1,
        )
    ]

    return data


def _confid_string_to_name(confid: pd.Series) -> pd.Series:
    confid = (
        confid.str.replace("vit_model", "vit")
        .str.replace("confidnet_", "")
        .str.replace("_dg", "_res")
        .str.replace("_det", "")
        .str.replace("det_", "")
        .str.replace("tcp", "confidnet")
        .str.upper()
        .str.replace("DEVRIES_DEVRIES", "DEVRIES")
        .str.replace("VIT_VIT", "VIT")
        .str.replace("DEVRIES", "Devries et al.")
        .str.replace("CONFIDNET", "ConfidNet")
        .str.replace("RES", "Res")
        .str.replace("_", "-")
        .str.replace("MCP", "MSR")
        .str.replace("VIT-Res", "VIT-DG-Res")
        .str.replace("VIT-DG-Res-", "VIT-DG-")
    )
    return confid


def rename_confids(data: pd.DataFrame) -> pd.DataFrame:
    """Encode model info in the confid name

    Args:
        data (pd.DataFrame): experiment data

    Returns:
        experiment data with renamed confids
    """
    data = data.assign(confid=_confid_string_to_name(data.model + "_" + data._confid))
    return data


def rename_studies(data: pd.DataFrame) -> pd.DataFrame:
    """Remove redundant info from study names.

    You should have transfered that info to other places using the other functions in this module
    beforehand.

    Args:
        data (pd.DataFrame): experiment data

    Returns:
        experiment data with renamed confids
    """
    data = data.assign(
        study=data.study.str.replace("tinyimagenet_384", "tinyimagenet_resize")
        .str.replace("vit", "")
        .str.replace("_384", "")
    )
    return data


def _filter_unused(data: pd.DataFrame) -> pd.DataFrame:
    data = data[
        (~data.confid.str.contains("waic"))
        & (~data.confid.str.contains("devries_mcd"))
        & (~data.confid.str.contains("devries_det"))
        & (~data.confid.str.contains("_sv"))
    ]
    return data


def str_format_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Format metrics to strings with appropriate precision

    Args:
        data (pd.DataFrame): experiment data

    Returns:
        experiment data with formatted metrics
    """
    data = data.rename(columns={"fail-NLL": "failNLL"})

    data = data.assign(
        accuracy=(data.accuracy * 100).map("{:>2.2f}".format),
        aurc=data.aurc.map("{:>3.2f}".format).map(
            lambda x: x[:4] if "." in x[:3] else x[:3]
        ),
        failauc=(data.failauc * 100).map("{:>3.2f}".format),
        ece=data.ece.map("{:>2.2f}".format),
        failNLL=data.failNLL.map("{:>2.2f}".format),
    )
    data = data.rename(columns={"failNLL": "fail-NLL"})

    return data


def main(base_path: str | Path):
    """Main entrypoint for CLI report generation

    Args:
        base_path (str | Path): path where experiment data lies
    """
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    data_dir: Path = Path(base_path).expanduser().resolve()
    data_dir.mkdir(exist_ok=True, parents=True)

    gather_data(data_dir)

    data, exp_names = load_data(data_dir)

    data = assign_hparams_from_names(data)

    data = filter_best_lr(data)
    data = filter_best_hparams(data)

    data = _filter_unused(data)
    data = rename_confids(data)
    data = rename_studies(data)

    plot_rank_style(data, "cifar10", "aurc", data_dir)
    vit_v_cnn_box(data, data_dir)

    data = tables.aggregate_over_runs(data)
    data = str_format_metrics(data)

    paper_results(data, "aurc", False, data_dir)
    paper_results(data, "aurc", False, data_dir, True)
    paper_results(data, "ece", False, data_dir)
    paper_results(data, "failauc", True, data_dir)
    paper_results(data, "accuracy", True, data_dir)
    paper_results(data, "fail-NLL", False, data_dir)

    rank_comparison_metric(data, data_dir)
    rank_comparison_mode(data, data_dir)
    rank_comparison_mode(data, data_dir, False)
