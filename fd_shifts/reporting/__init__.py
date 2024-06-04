import concurrent.futures
import functools
import os
from pathlib import Path
from typing import cast

import pandas as pd

from fd_shifts import logger
from fd_shifts.configs import Config
from fd_shifts.experiments import Experiment, get_all_experiments
from fd_shifts.experiments.configs import get_experiment_config, list_experiment_configs
from fd_shifts.experiments.tracker import list_analysis_output_files

DATASETS = (
    "svhn",
    "cifar10",
    "cifar100",
    "super_cifar100",
    "camelyon",
    "animals",
    "breeds",
)


def __find_in_store(config: Config, file: str) -> Path | None:
    store_paths = map(Path, os.getenv("FD_SHIFTS_STORE_PATH", "").split(":"))
    test_dir = config.test.dir.relative_to(os.getenv("EXPERIMENT_ROOT_DIR", ""))
    for store_path in store_paths:
        if (store_path / test_dir / file).is_file():
            logger.info(f"Loading {store_path / test_dir / file}")
            return store_path / test_dir / file


def __load_file(config: Config, name: str, file: str):
    if f := __find_in_store(config, file):
        return pd.read_csv(f)
    else:
        logger.error(f"Could not find {name}: {file} in store")
        return None


def __load_experiment(name: str) -> pd.DataFrame | None:
    from fd_shifts.main import omegaconf_resolve

    config = get_experiment_config(name)
    config = omegaconf_resolve(config)

    # data = list(executor.map(functools.partial(__load_file, config, name), list_analysis_output_files(config)))
    data = list(
        map(
            functools.partial(__load_file, config, name),
            list_analysis_output_files(config),
        )
    )
    if len(data) == 0 or any(map(lambda d: d is None, data)):
        return
    data = pd.concat(data)  # type: ignore
    data = (
        data.assign(
            experiment=config.data.dataset + ("vit" if "vit" in name else ""),
            run=int(name.split("run")[1].split("_")[0]),
            dropout=config.model.dropout_rate,
            rew=config.model.dg_reward if config.model.dg_reward is not None else 0,
            lr=config.trainer.optimizer.init_args["init_args"]["lr"],
        )
        .dropna(subset=["name", "model"])
        .drop_duplicates(subset=["name", "study", "model", "network", "confid"])
    )
    return data


def load_all():
    dataframes = []
    # TODO: make this async
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        dataframes = list(
            filter(
                lambda d: d is not None,
                executor.map(
                    __load_experiment,
                    list_experiment_configs(),
                ),
            )
        )

    data = pd.concat(dataframes)  # type: ignore
    data = data.loc[~data["study"].str.contains("tinyimagenet_original")]
    data = data.loc[~data["study"].str.contains("tinyimagenet_proposed")]

    # data = data.query(
    #     'not (experiment in ["cifar10", "cifar100", "super_cifar100"]'
    #     'and not name.str.contains("vgg13"))'
    # )

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

    return data


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
    logger.info("Assigning hyperparameters from experiment names")
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
        ).mask(
            data.backbone == "vit",
            data.name.str.split("model", expand=True)[1].str.split("_", expand=True)[0],
        ),
        # Encode every detail into confid name
        _confid=data.confid,
        confid=lambda data: data.model
        + "_"
        + data.confid
        + "_"
        + data.dropout.astype(str)
        + "_"
        + data.rew.astype(str),
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
    logger.info("Filtering best learning rates")

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

    logger.info("Filtering best hyperparameters")

    def _filter_row(row, selection_df, optimization_columns, fixed_columns):
        if "openset" in row["study"]:
            return True
        temp = selection_df[
            (row.experiment == selection_df.experiment)
            & (row._confid == selection_df._confid)
            & (row.model == selection_df.model)
        ]
        if len(temp) > 1:
            print(f"{len(temp)=}")
            raise ValueError("More than one row")

        if len(temp) == 0:
            return False

        temp = temp.iloc[0]

        result = row[optimization_columns] == temp[optimization_columns]
        # if result.all(axis=1).any().item():
        #     return True
        return result.all()

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
        .str.replace("medshifts/", "")
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


def main(out_path: str | Path):
    """Main entrypoint for CLI report generation

    Args:
        base_path (str | Path): path where experiment data lies
    """
    from fd_shifts.reporting import tables
    from fd_shifts.reporting.plots import plot_rank_style, vit_v_cnn_box
    from fd_shifts.reporting.tables import (
        paper_results,
        rank_comparison_metric,
        rank_comparison_mode,
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    data_dir: Path = Path(out_path).expanduser().resolve()
    data_dir.mkdir(exist_ok=True, parents=True)

    data = load_all()

    data = assign_hparams_from_names(data)

    data = filter_best_lr(data)
    data = filter_best_hparams(data)

    data = _filter_unused(data)
    data = rename_confids(data)
    data = rename_studies(data)

    # plot_rank_style(data, "cifar10", "aurc", data_dir)
    # vit_v_cnn_box(data, data_dir)

    data, std = tables.aggregate_over_runs(data)
    data = str_format_metrics(data)

    paper_results(data, "aurc", False, data_dir)
    # paper_results(data, "aurc", False, data_dir, rank_cols=True)
    # paper_results(data, "ece", False, data_dir)
    # paper_results(data, "failauc", True, data_dir)
    # paper_results(data, "accuracy", True, data_dir)
    # paper_results(data, "fail-NLL", False, data_dir)

    # rank_comparison_metric(data, data_dir)
    # rank_comparison_mode(data, data_dir)
    # rank_comparison_mode(data, data_dir, False)
