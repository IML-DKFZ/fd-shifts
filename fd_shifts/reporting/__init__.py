from pathlib import Path

import pandas as pd

from .tables import paper_results

# TODO: Refactor the rest
# TODO: Add error handling
# TODO: Implement sanity checks on final result table

# TODO: Take this from config
DATASETS = (
    "svhn",
    "cifar10",
    "cifar100",
    "super_cifar100",
    "camelyon",
    "animals",
    "breeds",
)


def load_file(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path)

    if not isinstance(result, pd.DataFrame):
        raise FileNotFoundError

    result = (
        result.assign(experiment=path.stem)
        .dropna(subset=["name", "model"])
        .drop_duplicates(subset=["name", "study", "model", "network", "confid"])
    )

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError

    return result


def load_data(data_dir: Path):
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
        'or experiment.str.contains("openset"))'
        'and not (study == "iid_study"))'
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

    data = data.assign(
        study=data.study.mask(
            data.experiment == "svhn_openset",
            "svhn_openset_study",
        )
    )

    data = data.assign(
        study=data.study.mask(
            data.experiment == "svhn_opensetvit",
            "svhnvit_openset_study",
        )
    )

    data = data.assign(
        study=data.study.mask(
            data.experiment == "animals_openset",
            "animals_openset_study",
        )
    )

    data = data.assign(
        study=data.study.mask(
            data.experiment == "animals_opensetvit",
            "animalsvit_openset_study",
        )
    )

    data = data.assign(ece=data.ece.mask(data.ece < 0))

    exp_names = list(
        filter(
            lambda exp: not exp.startswith("super_cifar100"),
            data.experiment.unique(),
        )
    )

    return data, exp_names


def extract_hparam(
    name: pd.Series, regex: str, default: str | None = None
) -> pd.Series:

    result: pd.Series = name.str.replace(".*" + regex + ".*", "\\1", regex=True)
    return result


def assign_hparams_from_names(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(
        backbone=lambda data: extract_hparam(data.name, r"bb([a-z0-9]+)(_small_conv)?"),
        # Prefix model name with vit_ if it is a vit model
        # If it isn't a vit model, model is the first part of the name
        model=lambda data: data["backbone"]
        .mask(data["backbone"] != "vit", "")
        .mask(data["backbone"] == "vit", "vit_")
        + data.model.where(
            data.backbone == "vit", data.name.str.split("_", expand=True)[0]
        ),
        run=lambda data: extract_hparam(data.name, r"run([0-9]+)"),
        dropout=lambda data: extract_hparam(data.name, r"do([01])"),
        rew=lambda data: extract_hparam(data.name, r"rew([0-9.]+)"),
        # Encode every detail into confid name
        # TODO: Should probably not be needed
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


def filter_best_hparams(data: pd.DataFrame, metric: str = "aurc") -> pd.DataFrame:
    """
    for every study (which encodes dataset) and confidence (which encodes other stuff)
    select all runs with the best avg combo of reward and dropout
    (maybe learning rate? should actually have been selected before)
    """

    def filter_row(row, selection_df, optimization_columns, fixed_columns):
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
    ]  # TODO: Merge these as soon as the first tuple doesn't encode everything anymore
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
            lambda row: filter_row(
                row, selection_df, optimization_columns, fixed_columns
            ),
            axis=1,
        )
    ]

    return data


def _confid_string_to_name(confid: pd.Series) -> pd.Series:
    confid = (
        confid.str.replace("confidnet_", "")
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
    data = data.assign(confid=_confid_string_to_name(data.model + "_" + data._confid))
    return data


def rename_studies(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(
        study=data.study.str.replace("tinyimagenet_384", "tinyimagenet_resize")
        .str.replace("vit", "")
        .str.replace("_384", "")
    )
    return data


def filter_unused(data: pd.DataFrame) -> pd.DataFrame:
    data = data[
        (~data.confid.str.contains("waic"))
        & (~data.confid.str.contains("devries_mcd"))
        & (~data.confid.str.contains("devries_det"))
        & (~data.confid.str.contains("_sv"))
        & (~data.confid.str.contains("_mi"))
    ]
    return data


def aggregate_over_runs(data: pd.DataFrame) -> pd.DataFrame:
    fixed_columns = ["study", "confid"]
    metrics_columns = ["accuracy", "aurc", "ece", "failauc", "fail-NLL"]

    data = (
        data[fixed_columns + metrics_columns]
        .groupby(by=fixed_columns)
        .mean()
        .sort_values("confid")
        .reset_index()
    )

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
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", -1)

    data_dir: Path = Path(base_path).expanduser().resolve()

    data, exp_names = load_data(data_dir)

    data = assign_hparams_from_names(data)

    data = filter_best_hparams(data)

    data = filter_unused(data)
    data = rename_confids(data)
    data = rename_studies(data)

    data = aggregate_over_runs(data)

    paper_results(data, "aurc", False, data_dir)
    paper_results(data, "ece", False, data_dir)
    paper_results(data, "failauc", True, data_dir)
    paper_results(data, "accuracy", True, data_dir)
    paper_results(data, "fail-NLL", False, data_dir)
