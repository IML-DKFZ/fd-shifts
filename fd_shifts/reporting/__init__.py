import concurrent.futures
import functools
import os
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel

from fd_shifts import logger
from fd_shifts.configs import Config
from fd_shifts.experiments.configs import get_experiment_config, list_experiment_configs
from fd_shifts.experiments.tracker import (
    list_analysis_output_files,
    list_bootstrap_analysis_output_files,
)

pandarallel.initialize(verbose=1)

DATASETS = (
    "svhn",
    "cifar10",
    "cifar100",
    "super_cifar100",
    "wilds_camelyon",
    "wilds_animals",
    "breeds",
)


def __find_in_store(config: Config, file: str) -> Path | None:
    store_paths = map(Path, os.getenv("FD_SHIFTS_STORE_PATH", "").split(":"))
    test_dir = config.test.dir.relative_to(os.getenv("EXPERIMENT_ROOT_DIR", ""))
    for store_path in store_paths:
        if (store_path / test_dir / file).is_file():
            # logger.info(f"Loading {store_path / test_dir / file}")
            return store_path / test_dir / file


def _load_file(config: Config, name: str, file: str):
    if f := __find_in_store(config, file):
        return pd.read_csv(f)
    else:
        logger.error(f"Could not find {name}: {file} in store")
        return None


def _load_experiment(
    name: str, bootstrap_analysis: bool = False
) -> pd.DataFrame | None:
    from fd_shifts.main import omegaconf_resolve

    config = get_experiment_config(name)
    config = omegaconf_resolve(config)

    # data = list(executor.map(functools.partial(_load_file, config, name), list_analysis_output_files(config)))
    if bootstrap_analysis:
        data = list(
            map(
                functools.partial(_load_file, config, name),
                list_bootstrap_analysis_output_files(config),
            )
        )
    else:
        data = list(
            map(
                functools.partial(_load_file, config, name),
                list_analysis_output_files(config),
            )
        )

    data = [d for d in data if d is not None]
    if len(data) == 0:
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
        .drop_duplicates(
            subset=(
                ["name", "study", "model", "network", "confid"]
                if not bootstrap_analysis
                else ["name", "study", "model", "network", "confid", "bootstrap_index"]
            )
        )
    )

    return data


def load_all(bootstrap_analysis: bool = False, include_vit: bool = True):
    dataframes = []
    # TODO: make this async
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        dataframes = list(
            filter(
                lambda d: d is not None,
                executor.map(
                    functools.partial(
                        _load_experiment, bootstrap_analysis=bootstrap_analysis
                    ),
                    filter(
                        (
                            (lambda exp: True)
                            if include_vit
                            else lambda exp: not exp.startswith("vit")
                        ),
                        list_experiment_configs(),
                    ),
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
    logger.info(f"Filtering best learning rates, optimizing {metric}")

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

    return data, selection_df


def filter_best_hparams(
    data: pd.DataFrame, metric: str = "aurc", bootstrap_analysis: bool = False
) -> pd.DataFrame:
    """
    for every study (which encodes dataset) and confidence (which encodes other stuff)
    select all runs with the best avg combo of reward and dropout

    Args:
        data (pd.DataFrame): experiment data
        metric (str): metric to select best from

    Returns:
        filtered data
    """
    logger.info(f"Filtering best hyperparameters, optimizing {metric}")

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
    if bootstrap_analysis:
        aggregation_columns = ["run", "bootstrap_index", metric]
    else:
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
        data.parallel_apply(
            lambda row: _filter_row(
                row, selection_df, optimization_columns, fixed_columns
            ),
            axis=1,
        )
    ]

    return data, selection_df


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
    _columns = data.columns
    dash_to_no_dash = {
        c: c.replace("-", "") for c in _columns if isinstance(c, str) and "-" in c
    }
    # Remove dashes from column names
    data = data.rename(columns=dash_to_no_dash)

    # Formatting instructions for each metric
    format_mapping = {
        "accuracy": lambda x: "{:>2.2f}".format(x * 100),
        "aurc": lambda x: (
            "{:>3.2f}".format(x)[:4]
            if "." in "{:>3.2f}".format(x)[:3]
            else "{:>3.2f}".format(x)[:3]
        ),
        "failauc": lambda x: "{:>3.2f}".format(x * 100),
        "ece": lambda x: "{:>2.2f}".format(x),
        "failNLL": lambda x: "{:>2.2f}".format(x),
    }
    format_mapping["eaurc"] = format_mapping["aurc"]
    format_mapping["augrc"] = format_mapping["aurc"]
    format_mapping["eaugrc"] = format_mapping["aurc"]
    format_mapping["aurcba"] = format_mapping["aurc"]
    format_mapping["augrcba"] = format_mapping["aurc"]

    # Apply formatting if metric is present in the data
    for col, formatting_func in format_mapping.items():
        if col in data.columns:
            data[col] = data[col].map(formatting_func)

    # Apply inverse mapping, add dashes again
    data = data.rename(columns={v: k for k, v in dash_to_no_dash.items()})

    return data


def main(
    out_path: str | Path = "./output",
    metric_hparam_search: str = "augrc",
):
    """Main entrypoint for CLI report generation"""
    from fd_shifts.reporting import tables
    from fd_shifts.reporting.plots import plot_rank_style, vit_v_cnn_box
    from fd_shifts.reporting.tables import paper_results, rank_comparison_metric

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    data_dir: Path = Path(out_path).expanduser().resolve()
    data_dir = data_dir / f"optimized-{metric_hparam_search}"
    data_dir.mkdir(exist_ok=True, parents=True)

    data = load_all()
    data = assign_hparams_from_names(data)

    # -- Select best hyperparameters ---------------------------------------------------
    data, selection_df = filter_best_lr(data, metric=metric_hparam_search)
    selection_df.to_csv(data_dir / "filter_best_lr.csv", decimal=".")
    logger.info(f"Saved best lr to '{str(data_dir / 'filter_best_lr.csv')}'")
    data, selection_df = filter_best_hparams(data, metric=metric_hparam_search)
    selection_df.to_csv(data_dir / "filter_best_hparams.csv", decimal=".")
    logger.info(f"Saved best hparams to '{str(data_dir / 'filter_best_hparams.csv')}'")

    data = _filter_unused(data)

    # Filter MCD data
    # data = data[~data.confid.str.contains("mcd")]

    data = rename_confids(data)
    data = rename_studies(data)

    # -- Aggregate across runs ---------------------------------------------------------
    data, std = tables.aggregate_over_runs(
        data,
        metric_columns=[
            "accuracy",
            "aurc",
            "ece",
            "failauc",
            "fail-NLL",
            "e-aurc",
            "augrc",
            "e-augrc",
            "aurc-ba",
            "augrc-ba",
        ],
    )

    # -- Apply metric formatting -------------------------------------------------------
    data = str_format_metrics(data)

    # # -- Relative error (evaluated across runs) --------------------------------------
    metric_list = ["aurc", "e-aurc", "augrc", "e-augrc", "aurc-ba", "augrc-ba"]

    # data_dir_std = data_dir / "rel_std"
    # data_dir_std.mkdir(exist_ok=True, parents=True)
    # for m in metric_list:
    #     std[m] = std[m].astype(float) / data[m].astype(float)
    # std = str_format_metrics(std)

    # for m in metric_list:
    #     # lower is better for all these metrics
    #     paper_results(std, m, False, data_dir_std)

    # # -- Metric tables -----------------------------------------------------------------
    for m in metric_list:
        # lower is better for all these metrics
        paper_results(data, m, False, data_dir)
        paper_results(data, m, False, data_dir, rank_cols=True)

    paper_results(data, "ece", False, data_dir)
    paper_results(data, "failauc", True, data_dir)
    paper_results(data, "accuracy", True, data_dir)
    paper_results(data, "fail-NLL", False, data_dir)

    # -- Ranking comparisons -----------------------------------------------------------
    rank_comparison_metric(
        data,
        data_dir,
        metric1="aurc",
        metric2="augrc",
        metric1_higherbetter=False,
        metric2_higherbetter=False,
    )
