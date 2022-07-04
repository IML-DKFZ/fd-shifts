import gc
import re
from itertools import zip_longest
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display
from rich import print

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

    metric = "aurc"
    def _aggregate_over_runs(df):
        non_agg_columns = ["study", "confid"]  # might need rew if no model selection
        filter_metrics_df = df[non_agg_columns + ["run", metric]]
        df_mean = (
            filter_metrics_df.groupby(by=non_agg_columns).mean().reset_index().round(2)
        )
        df_std = filter_metrics_df.groupby(by=non_agg_columns).std().reset_index().round(2)

        studies = df_mean.study.unique().tolist()
        dff = pd.DataFrame({"confid": df.confid.unique()})
        #     print(dff)
        #     print("CHECK LEN DFF", len(dff), len(df_mean))
        combine_and_str = False
        if combine_and_str:
            agg_mean_std = (
                lambda s1, s2: s1
                if (s1.name == "confid" or s1.name == "study" or s1.name == "rew")
                else s1.astype(str) + " ± " + s2.astype(str)
            )
            df_mean = df_mean.combine(df_std, agg_mean_std)
            for s in studies:
                sdf = df_mean[df_mean.study == s]
                dff[s] = dff["confid"].map(sdf.set_index("confid")[metric])

        else:
            for s in studies:
                sdf = df_mean[df_mean.study == s]
                dff[s] = dff["confid"].map(sdf.set_index("confid")[metric])
                # print("DFF", dff.columns.tolist())

        return dff
    # dff = _aggregate_over_runs(data)


    def paper_results(df, metric, invert):
        non_agg_columns = ["study", "confid"]  # might need rew if no model selection
        df_acc = (
            df[non_agg_columns + ["run", "accuracy"]]
            .groupby(by=non_agg_columns)
            .mean()
            .sort_values("confid")
            .reset_index()
            #         .round(2)
        )
        df_aurc = (
            df[non_agg_columns + ["run", "aurc"]]
            .groupby(by=non_agg_columns)
            .mean()
            .sort_values("confid")
            .reset_index()
        )
        df_auc = (
            df[non_agg_columns + ["run", "failauc"]]
            .groupby(by=non_agg_columns)
            .mean()
            .sort_values("confid")
            .reset_index()
        )
        df_ece = (
            df[non_agg_columns + ["run", "ece"]]
            .groupby(by=non_agg_columns)
            .mean()
            .sort_values("confid")
            .reset_index()
        )
        df_nll = (
            df[non_agg_columns + ["run", "fail-NLL"]]
            .groupby(by=non_agg_columns)
            .mean()
            .sort_values("confid")
            .reset_index()
        )
        df_acc["accuracy"] = df_acc["accuracy"] * 100
        df_acc["accuracy"] = df_acc["accuracy"].map("{:>2.2f}".format)

        df_aurc["accuracy"] = df_aurc["aurc"]
        #     df_aurc = df_aurc.round(2)
        df_acc["aurc"] = (
            df_aurc["aurc"]
            .map("{:>3.2f}".format)
            .map(lambda x: x[:4] if "." in x[:3] else x[:3])
        )
        df_aurc["accuracy"] = df_aurc["accuracy"].map("{:>3.2f}".format)

        df_auc["accuracy"] = df_auc["failauc"] * 100
        #     df_auc = df_auc.round(2)
        df_acc["failauc"] = (df_auc["failauc"] * 100).map("{:>3.2f}".format)
        df_auc["accuracy"] = df_auc["accuracy"].map("{:>2.2f}".format)

        df_ece["accuracy"] = df_ece["ece"]
        #     df_ece = df_ece.round(2)
        df_acc["ece"] = df_ece["ece"].map("{:>2.2f}".format)
        df_ece["accuracy"] = df_ece["accuracy"].map("{:>2.2f}".format)

        df_nll["accuracy"] = df_nll["fail-NLL"]
        #     df_nll = df_nll.round(2)
        df_acc["fail-NLL"] = df_nll["fail-NLL"].map("{:>2.2f}".format)
        df_nll["accuracy"] = df_nll["accuracy"].map("{:>2.2f}".format)

        studies = df_acc.study.unique().tolist()

        paper_dff = df_acc[["confid", "study", metric]]
        paper_dff = paper_dff[~paper_dff.study.str.contains("val_tuning")]
        paper_dff = paper_dff[~paper_dff.study.str.contains("original")]
        paper_dff = (
            pd.pivot(paper_dff, index="confid", columns="study")
            .swaplevel(0, 1, 1)
            .sort_index(axis=1, level=0)
            .reset_index()
            .assign(
                classifier=lambda row: row.confid.where(
                    row.confid.str.contains("VIT"), "CNN"
                )
            )
        )
        paper_dff.loc[paper_dff.classifier.str.contains("VIT"), "classifier"] = "ViT"
        #     print(paper_dff.columns)
        paper_dff[("cifar10_noise_study", metric)] = (
            paper_dff[
                paper_dff.columns[
                    paper_dff.columns.get_level_values(0).str.startswith("cifar10_")
                    & paper_dff.columns.get_level_values(0).str.contains("noise")
                ]
            ]
            .astype(float)
            .mean(axis=1)
            .reindex(paper_dff.index)
        )
        paper_dff[("cifar100_noise_study", metric)] = (
            paper_dff[
                paper_dff.columns[
                    paper_dff.columns.get_level_values(0).str.startswith("cifar100_")
                    & paper_dff.columns.get_level_values(0).str.contains("noise")
                ]
            ]
            .astype(float)
            .mean(axis=1)
            .reindex(paper_dff.index)
        )

        paper_dff = paper_dff[
            paper_dff.columns[
                ~paper_dff.columns.get_level_values(0).str.contains("noise_study_")
            ]
        ].sort_index(axis=1, level=0)

        def rename_study(s):
            if s in ["confid", "classifier"]:
                return (s, "", "")

            return (
                s.split("_")[0],
                s.split("_")[1]
                .replace("in", "sub")
                .replace(
                    "new",
                    "s-ncs"
                    if "cifar" in s.split("_")[0]
                    and "cifar" in "".join(s.split("_")[1:])
                    else "ns-ncs",
                )
                .replace("openset", "s-ncs")
                .replace("noise", "cor"),
                s.split("_")[4].replace("tinyimagenet", "ti").replace("cifar", "c")
                if "new" in s
                else "",
            )

        paper_dff.columns = paper_dff.columns.get_level_values(0).map(rename_study)
        paper_dff.confid = paper_dff.confid.str.replace("VIT-", "")

        paper_dff = paper_dff[
            paper_dff.confid.isin(
                [
                    "ConfidNet",
                    "DG-MCD-EE",
                    "DG-Res",
                    "Devries et al.",
                    "MCD-EE",
                    "MCD-MSR",
                    "MCD-PE",
                    "MSR",
                    "PE",
                    "MAHA",
                ]
            )
        ]
        # paper_dff = paper_dff[~(paper_dff.confid.isin(["ConfidNet", "Devries et al.", "DG-MCD-EE", "DG-Res",]) & paper_dff.classifier.str.contains("ViT"))]
        paper_dff = paper_dff.sort_values(["classifier", "confid"]).set_index(
            ["confid", "classifier"]
        )

        paper_dff.index = paper_dff.index.set_names([None, None])
        paper_dff.columns = paper_dff.columns.set_names(["", "study", "ncs-data set"])

        columns = [
            ("animals", "iid", ""),
            ("animals", "sub", ""),
            ("animals", "s-ncs", ""),
            ("breeds", "iid", ""),
            ("breeds", "sub", ""),
        ]
        paper_dff = paper_dff[
            [
                ("animals", "iid", ""),
                ("animals", "sub", ""),
                ("animals", "s-ncs", ""),
                ("breeds", "iid", ""),
                ("breeds", "sub", ""),
                ("camelyon", "iid", ""),
                ("camelyon", "sub", ""),
                ("cifar100", "iid", ""),
                ("cifar100", "sub", ""),
                ("cifar100", "cor", ""),
                ("cifar100", "s-ncs", "c10"),
                ("cifar100", "ns-ncs", "svhn"),
                ("cifar100", "ns-ncs", "ti"),
                ("cifar10", "iid", ""),
                ("cifar10", "cor", ""),
                ("cifar10", "s-ncs", "c100"),
                ("cifar10", "ns-ncs", "svhn"),
                ("cifar10", "ns-ncs", "ti"),
                ("svhn", "iid", ""),
                ("svhn", "s-ncs", ""),
                ("svhn", "ns-ncs", "c10"),
                ("svhn", "ns-ncs", "c100"),
                ("svhn", "ns-ncs", "ti"),
            ]
        ].rename(
            columns={
                "animals": "iWildCam",
                "breeds": "BREEDS",
                "camelyon": "CAMELYON",
                "cifar10": "CIFAR-10",
                "cifar100": "CIFAR-100",
                "svhn": "SVHN",
            },
            level=0,
        )

        #     paper_dff.loc[("DG-MCD-EE", "VIT"), ("breeds", "iid", "")] = np.nan
        #     paper_dff.loc[("DG-Res", "VIT"), ("breeds", "iid", "")] = np.nan
        #     paper_dff.loc[("DG-MCD-EE", "VIT"), ("breeds", "sub", "")] = np.nan
        #     paper_dff.loc[("DG-Res", "VIT"), ("breeds", "sub", "")] = np.nan

        #     paper_dff.loc[("DG-MCD-EE", "VIT"), ("camelyon", "iid", "")] = np.nan
        #     paper_dff.loc[("DG-Res", "VIT"), ("camelyon", "iid", "")] = np.nan
        #     paper_dff.loc[("DG-MCD-EE", "VIT"), ("camelyon", "sub", "")] = np.nan
        #     paper_dff.loc[("DG-Res", "VIT"), ("camelyon", "sub", "")] = np.nan

        #     paper_dff.loc[("DG-MCD-EE", "VIT"), ("animals", "iid", "")] = np.nan
        #     paper_dff.loc[("DG-Res", "VIT"), ("animals", "iid", "")] = np.nan
        #     paper_dff.loc[("DG-MCD-EE", "VIT"), ("animals", "sub", "")] = np.nan
        #     paper_dff.loc[("DG-Res", "VIT"), ("animals", "sub", "")] = np.nan
        #     paper_dff.loc[("DG-MCD-EE", "VIT"), ("animals", "s-ncs", "")] = np.nan
        #     paper_dff.loc[("DG-Res", "VIT"), ("animals", "s-ncs", "")] = np.nan
        cmap = "Oranges_r" if invert else "Oranges"

        ltex = (
            paper_dff.astype(float)
            .style.background_gradient(
                cmap,
                subset=(
                    paper_dff.index[
                        paper_dff.index.get_level_values(1).str.contains("ViT")
                    ],
                    paper_dff.columns,
                ),
            )
            .background_gradient(
                cmap,
                subset=(
                    paper_dff.index[
                        ~paper_dff.index.get_level_values(1).str.contains("ViT")
                    ],
                    paper_dff.columns,
                ),
            )
            .highlight_null(props="background-color: white;color: black")
            .format(
                lambda x: f"{x:>3.2f}"[:4]
                if "." in f"{x:>3.2f}"[:3]
                else f"{x:>3.2f}"[:3],
                na_rep="*",
            )
        )

        display(HTML(f"<h2>{metric}</h2>"))
        display(HTML(ltex.to_html()))

        with open(data_dir / f"{metric}_now.csv", "w") as f:
            # f.write(ltex.data[((~ltex.data.index.get_level_values(0).str.contains("DG")) & (~ltex.data.index.get_level_values(0).str.contains("Devries"))) | (~ltex.data.index.get_level_values(1).str.contains("ViT"))].applymap(lambda x: f"{x:>3.2f}"[:4] if "." in f"{x:>3.2f}"[:3] else f"{x:>3.2f}"[:3]).to_csv())
            f.write(
                ltex.data.applymap(
                    lambda x: f"{x:>3.2f}"[:4]
                    if "." in f"{x:>3.2f}"[:3]
                    else f"{x:>3.2f}"[:3]
                ).to_csv()
            )

        ltex.data.columns = ltex.data.columns.set_names(
            ["\\multicolumn{1}{c}{}", "study", "ncs-data set"]
        )
        ltex = ltex.to_latex(
            convert_css=True,
            hrules=True,
            multicol_align="c?",
            column_format="ll?rrr?rr?rr?rrrrrr?rrrrr?rrrrr",
        )

        ltex = ltex.split("\n")
        del ltex[1]
        # del ltex[4]
        ltex[1] = ltex[1].replace("?", "")
        ltex[2] = ltex[2][: ltex[2].rfind("?")] + ltex[2][ltex[2].rfind("?") + 1 :]
        i = ltex.index(next((x for x in ltex if "ViT" in x)))
        ltex.insert(i, "\\midrule \\\\")
        ltex = "\n".join(ltex)

        with open(data_dir / f"paper_results_{metric}.tex", "w") as f:
            f.write(ltex)

    paper_results(data, "aurc", False)
    paper_results(data, "ece", False)
    paper_results(data, "failauc", True)
    paper_results(data, "accuracy", True)
    paper_results(data, "fail-NLL", False)
    # print(paper_dff)
    gc.collect()
