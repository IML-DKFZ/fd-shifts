from pathlib import Path
from typing import Callable

import pandas as pd

_sanity_checks: dict[str, list[Callable]] = {}


def register_sanity_check(metric: str):
    def _inner_func(func: Callable) -> Callable:
        if not metric in _sanity_checks:
            _sanity_checks[metric] = []
        _sanity_checks[metric].append(func)
        return func

    return _inner_func


# @register_sanity_check("aurc")
def check_binary_class_msr_pe_equal(table: pd.DataFrame):
    # TODO: Broken for current test data
    return (
        table.loc[("MSR", "CNN"), ("CAMELYON", "iid", "")]
        == table.loc[("PE", "CNN"), ("CAMELYON", "iid", "")]
    ) and (
        table.loc[("MSR", "ViT"), ("CAMELYON", "iid", "")]
        == table.loc[("PE", "ViT"), ("CAMELYON", "iid", "")]
    )


def sanity_check(table: pd.DataFrame, metric: str):
    if not metric in _sanity_checks:
        return

    for check in _sanity_checks[metric]:
        assert check(table)


def _create_results_pivot(data: pd.DataFrame, metric: str):
    results_table = data[["confid", "study", metric]]

    # Remove val_tuning and original mode studies
    results_table = results_table[~results_table.study.str.contains("val_tuning")]
    results_table = results_table[~results_table.study.str.contains("original")]

    results_table = (
        pd.pivot(results_table, index="confid", columns="study")
        .swaplevel(0, 1, 1)
        .sort_index(axis=1, level=0)
        .reset_index()
        .assign(
            classifier=lambda row: row.confid.where(
                row.confid.str.contains("VIT"), "CNN"
            ).mask(row.confid.str.contains("VIT"), "ViT")
        )
    )
    return results_table


def _aggregate_noise_studies(data: pd.DataFrame, metric: str):
    data[("cifar10_noise_study", metric)] = (
        data[
            data.columns[
                data.columns.get_level_values(0).str.startswith("cifar10_")
                & data.columns.get_level_values(0).str.contains("noise")
            ]
        ]
        .astype(float)
        .mean(axis=1)
        .reindex(data.index)
    )
    data[("cifar100_noise_study", metric)] = (
        data[
            data.columns[
                data.columns.get_level_values(0).str.startswith("cifar100_")
                & data.columns.get_level_values(0).str.contains("noise")
            ]
        ]
        .astype(float)
        .mean(axis=1)
        .reindex(data.index)
    )
    data = data[
        data.columns[~data.columns.get_level_values(0).str.contains("noise_study_")]
    ].sort_index(axis=1, level=0)

    return data


def _study_name_to_multilabel(study_name):
    if study_name in ["confid", "classifier"]:
        return (study_name, "", "")

    return (
        study_name.split("_")[0],
        study_name.split("_")[1]
        .replace("in", "sub")
        .replace(
            "new",
            "s-ncs"
            if "cifar" in study_name.split("_")[0]
            and "cifar" in "".join(study_name.split("_")[1:])
            else "ns-ncs",
        )
        .replace("openset", "s-ncs")
        .replace("noise", "cor"),
        study_name.split("_")[4].replace("tinyimagenet", "ti").replace("cifar", "c")
        if "new" in study_name
        else "",
    )


_study_list = [
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


def _reorder_studies(table: pd.DataFrame) -> pd.DataFrame:
    """Reorder studies that are in a well-defind order for publication, append others to the end"""
    ordered_columns = [
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

    all_columns = list(table.columns)
    ordered_columns = list(filter(lambda c: c in all_columns, ordered_columns)) + list(
        filter(lambda c: c not in ordered_columns, all_columns)
    )
    table = table[ordered_columns]
    return table


def _dataset_to_display_name(dataset_name: str) -> str:
    mapping = {
        "animals": "iWildCam",
        "breeds": "BREEDS",
        "camelyon": "CAMELYON",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "svhn": "SVHN",
    }
    return mapping[dataset_name]


def _build_multilabel(table: pd.DataFrame) -> pd.DataFrame:
    table.columns = table.columns.get_level_values(0).map(_study_name_to_multilabel)
    table.confid = table.confid.str.replace("VIT-", "")

    table = table[table.confid.isin(_study_list)]
    table = table.sort_values(["classifier", "confid"]).set_index(
        ["confid", "classifier"]
    )

    table.index = table.index.set_names([None, None])
    table.columns = table.columns.set_names(["", "study", "ncs-data set"])

    return table


def paper_results(data: pd.DataFrame, metric: str, invert: bool, out_dir: Path):
    results_table = _create_results_pivot(data, metric)
    results_table = _aggregate_noise_studies(results_table, metric)

    results_table = _build_multilabel(results_table)
    results_table = _reorder_studies(results_table)

    results_table = results_table.rename(
        columns=_dataset_to_display_name,
        level=0,
    )

    sanity_check(results_table, metric)

    # Render table
    cmap = "Oranges_r" if invert else "Oranges"

    ltex = (
        results_table.astype(float)
        .style.background_gradient(
            cmap,
            subset=(
                results_table.index[
                    results_table.index.get_level_values(1).str.contains("ViT")
                ],
                results_table.columns,
            ),
        )
        .background_gradient(
            cmap,
            subset=(
                results_table.index[
                    ~results_table.index.get_level_values(1).str.contains("ViT")
                ],
                results_table.columns,
            ),
        )
        .highlight_null(props="background-color: white;color: black")
        .format(
            lambda x: f"{x:>3.2f}"[:4] if "." in f"{x:>3.2f}"[:3] else f"{x:>3.2f}"[:3],
            na_rep="*",
        )
    )

    # display(HTML(f"<h2>{metric}</h2>"))
    # display(HTML(ltex.to_html()))

    with open(out_dir / f"{metric}_now.csv", "w") as f:
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

    with open(out_dir / f"paper_results_{metric}.tex", "w") as f:
        f.write(ltex)
