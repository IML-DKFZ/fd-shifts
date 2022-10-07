import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

LATEX_TABLE_TEMPLATE = r"""
\documentclass{article} % For LaTeX2e
\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{bm}
\usepackage{multirow}

\begin{document}

\begin{table*}%[h!]
\centering
\caption{\textbf{FD-Shifts Benchmark Results measured as {metric}}}
\label{tab:results}
\vspace{0.3cm}
\scalebox{0.265}{
\newcolumntype{?}{!{\vrule width 2pt}}
\renewcommand{\arraystretch}{1.5}
\huge
\input{{input_file}}}
\end{table*}
\end{document}
"""


_sanity_checks: dict[str, list[Callable]] = {}


def register_sanity_check(metric: str):
    def _inner_func(func: Callable) -> Callable:
        if not metric in _sanity_checks:
            _sanity_checks[metric] = []
        _sanity_checks[metric].append(func)
        return func

    return _inner_func


def str_to_float(x: str):
    try:
        return float(x)
    except:
        return None


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


# @register_sanity_check("accuracy")
def check_accuracy_valid_range(table: pd.DataFrame):
    tmp = table.applymap(str_to_float)
    return ((tmp > 0) & (tmp < 100)).all()


# @register_sanity_check("aurc")
def check_aurc_valid_range(table: pd.DataFrame):
    tmp = table.applymap(str_to_float)
    return ((tmp > 0) & (tmp < 1000)).all()


# @register_sanity_check("ece")
def check_ece_valid_range(table: pd.DataFrame):
    tmp = table.applymap(str_to_float)
    return ((tmp > 0) & (tmp < 1)).all()


def sanity_check(table: pd.DataFrame, metric: str):
    if not metric in _sanity_checks:
        return

    for check in _sanity_checks[metric]:
        assert check(table)


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
    return data


def _create_results_pivot(data: pd.DataFrame, metric: str, original_mode: bool = False):
    results_table = data[["confid", "study", metric]]

    # Remove val_tuning and original mode studies
    results_table = results_table[~results_table.study.str.contains("val_tuning")]
    if original_mode:
        results_table = results_table[~results_table.study.str.contains("proposed")]
    else:
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
    "MSR",
    "MLS",
    "PE",
    "MCD-MSR",
    "MCD-MLS",
    "MCD-PE",
    "MCD-EE",
    "MCD-MI",
    "ConfidNet",
    "DG-MCD-MSR",
    "DG-Res",
    "Devries et al.",
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


def _reorder_confids(data: pd.DataFrame) -> pd.DataFrame:
    data = data.reindex(labels=_study_list, level=0).sort_index(
        level=1, sort_remaining=False
    )
    # print(data)
    return data


def build_results_table(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    results_table = _create_results_pivot(data, metric)
    results_table = _aggregate_noise_studies(results_table, metric)

    results_table = _build_multilabel(results_table)
    results_table = _reorder_studies(results_table)
    results_table = _reorder_confids(results_table)
    return results_table


def _print_original_mode(data: pd.DataFrame, metric: str):
    results_table = _create_results_pivot(data, metric, original_mode=True)
    results_table = _aggregate_noise_studies(results_table, metric)

    results_table = _build_multilabel(results_table)
    results_table = _reorder_studies(results_table)
    results_table = _reorder_confids(results_table)
    print(f"{metric} original mode")
    print(results_table)


def _compute_gmap(data: pd.DataFrame, invert: bool):
    # NOTE: Manually compute gradient map because Normalize returns 0 if vmax - vmin == 0, but we
    # NOTE:   want it to be 1 in that case

    gmap = data.to_numpy(float)
    gmap_min = np.nanmin(gmap, axis=0)
    gmap_max = np.nanmax(gmap, axis=0)

    for col in range(gmap.shape[1]):
        vmin = gmap_min[col] - (0.0001 if invert else 0)
        vmax = gmap_max[col] + (0 if invert else 0.0001)
        gmap[:, col] = matplotlib.colors.Normalize(vmin, vmax)(gmap[:, col])

    return gmap


def paper_results(data: pd.DataFrame, metric: str, invert: bool, out_dir: Path):
    _print_original_mode(data, metric)
    results_table = build_results_table(data, metric)

    results_table = results_table.rename(
        columns=_dataset_to_display_name,
        level=0,
    )
    # print(results_table.astype(float))

    sanity_check(results_table, metric)

    # Render table
    cmap = "Oranges_r" if invert else "Oranges"
    results_table = results_table.astype(float).applymap(
        lambda val: round(val, 2)
        if val < 10
        else round(val, 1)
        # lambda val: round(val, 4) if val < 10 else round(val, 3)
    )

    gmap_vit = _compute_gmap(
        results_table.loc[
            results_table.index[
                results_table.index.get_level_values(1).str.contains("ViT")
            ],
            results_table.columns,
        ],
        invert,
    )
    gmap_cnn = _compute_gmap(
        results_table.loc[
            results_table.index[
                ~results_table.index.get_level_values(1).str.contains("ViT")
            ],
            results_table.columns,
        ],
        invert,
    )

    ltex = (
        results_table.style.background_gradient(
            cmap,
            axis=None,
            subset=(
                results_table.index[
                    results_table.index.get_level_values(1).str.contains("ViT")
                ],
                results_table.columns,
            ),
            gmap=gmap_vit,
        )
        .background_gradient(
            cmap,
            axis=None,
            subset=(
                results_table.index[
                    ~results_table.index.get_level_values(1).str.contains("ViT")
                ],
                results_table.columns,
            ),
            gmap=gmap_cnn,
        )
        .highlight_null(props="background-color: white;color: black")
        .format(
            lambda x: f"{x:>3.2f}"[:4] if "." in f"{x:>3.2f}"[:3] else f"{x:>3.2f}"[:3],
            na_rep="*",
        )
        # .format(
        #     lambda x: f"{x:>3.4f}"[:6] if "." in f"{x:>3.4f}"[:5] else f"{x:>3.4f}"[:5],
        #     na_rep="*",
        # )
    )

    # with open(out_dir / f"paper_results_{metric}.csv", "w") as f:
    #     f.write(
    #         ltex.data.applymap(
    #             lambda x: f"{x:>3.2f}"[:4]
    #             if "." in f"{x:>3.2f}"[:3]
    #             else f"{x:>3.2f}"[:3]
    #         ).to_csv()
    #     )
    # with open(out_dir / f"paper_results_{metric}.html", "w") as f:
    #     f.write(
    #         ltex.set_table_styles(
    #             [
    #                 {
    #                     "selector": "th",
    #                     "props": [
    #                         ("font-size", "16pt"),
    #                         ("border-style", "solid"),
    #                         ("border-width", "2px"),
    #                         ("margin", "0px"),
    #                         ("border-collapse", "collapse"),
    #                         ("border-spacing", "0"),
    #                     ],
    #                 },
    #                 {
    #                     "selector": "td",
    #                     "props": [
    #                         ("font-size", "16pt"),
    #                         ("border", "1px solid black"),
    #                         ("margin", "0px"),
    #                         ("border-collapse", "collapse"),
    #                         ("border-spacing", "0"),
    #                     ],
    #                 },
    #                 {
    #                     "selector": "thead",
    #                     "props": [
    #                         ("font-size", "16pt"),
    #                         ("border", "1px solid black"),
    #                         ("margin", "0px"),
    #                         ("border-collapse", "collapse"),
    #                         ("border-spacing", "0"),
    #                     ],
    #                 },
    #                 {
    #                     "selector": "tbody",
    #                     "props": [
    #                         ("font-size", "16pt"),
    #                         ("border", "1px solid black"),
    #                         ("margin", "0px"),
    #                         ("border-collapse", "collapse"),
    #                         ("border-spacing", "0"),
    #                     ],
    #                 },
    #                 {
    #                     "selector": "",
    #                     "props": [
    #                         ("font-size", "16pt"),
    #                         ("border", "1px solid black"),
    #                         ("margin", "0px"),
    #                         ("border-collapse", "collapse"),
    #                         ("border-spacing", "0"),
    #                     ],
    #                 },
    #             ]
    #         )
    #         .set_properties(
    #             **{
    #                 "font-size": "16pt",
    #                 "font-family": "Victor Mono",
    #                 "padding": "1rem",
    #                 "border": "1px solid black",
    #                 "margin": "0px",
    #                 "border-collapse": "collapse",
    #                 "border-spacing": "0",
    #                 # "background-color": "white",
    #                 # "color": "black",
    #                 "text-align": "center",
    #             }
    #         )
    #         .set_table_styles(
    #             {
    #                 ("iWildCam", "iid", "",): [
    #                     {"selector": "th", "props": "border-left: 4px solid black"},
    #                     {"selector": "td", "props": "border-left: 4px solid black"},
    #                 ],
    #                 ("BREEDS", "iid", "",): [
    #                     {"selector": "th", "props": "border-left: 4px solid black"},
    #                     {"selector": "td", "props": "border-left: 4px solid black"},
    #                 ],
    #                 ("CAMELYON", "iid", "",): [
    #                     {"selector": "th", "props": "border-left: 4px solid black"},
    #                     {"selector": "td", "props": "border-left: 4px solid black"},
    #                 ],
    #                 ("CIFAR-100", "iid", "",): [
    #                     {"selector": "th", "props": "border-left: 4px solid black"},
    #                     {"selector": "td", "props": "border-left: 4px solid black"},
    #                 ],
    #                 ("CIFAR-10", "iid", "",): [
    #                     {"selector": "th", "props": "border-left: 4px solid black"},
    #                     {"selector": "td", "props": "border-left: 4px solid black"},
    #                 ],
    #                 ("SVHN", "iid", "",): [
    #                     {"selector": "th", "props": "border-left: 4px solid black"},
    #                     {"selector": "td", "props": "border-left: 4px solid black"},
    #                 ],
    #             },
    #             overwrite=False,
    #             axis=0,
    #         )
    #         .set_table_styles(
    #             {
    #                 ("PE", "CNN",): [
    #                     {"selector": "tr", "props": "border-spacing: 1rem"},
    #                     {"selector": "th", "props": "border-spacing: 1rem"},
    #                     {"selector": "td", "props": "border-spacing: 1rem"},
    #                 ],
    #             },
    #             overwrite=False,
    #             axis=1,
    #         )
    #         .to_html()
    #     )

    ltex.data.columns = ltex.data.columns.set_names(
        ["\\multicolumn{1}{c}{}", "study", "ncs-data set"]
    )
    ltex = ltex.to_latex(
        convert_css=True,
        hrules=True,
        multicol_align="c?",
        column_format="ll?rrr?rr?rr?rrrrrr?rrrrr?rrrrr",
    )

    # Remove toprule
    ltex = list(filter(lambda line: line != r"\toprule", ltex.splitlines()))

    # No separators in first header row
    ltex[1] = ltex[1].replace("?", "")

    # Remove last separator in second header row
    # (this is just `replace("?", "", 1)`, but from the right)
    ltex[2] = ltex[2][: ltex[2].rfind("?")] + ltex[2][ltex[2].rfind("?") + 1 :]

    # Insert empty row before ViT part
    i = ltex.index(next((x for x in ltex if "ViT" in x)))
    ltex.insert(i, "\\midrule \\\\")

    ltex = "\n".join(ltex)

    with open(out_dir / f"paper_results_{metric}.tex", "w") as f:
        f.write(ltex)

    # TODO: Make this toggleable
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy2(
            out_dir / f"paper_results_{metric}.tex",
            tmpdir / f"paper_results_{metric}.tex",
        )
        with open(tmpdir / "render.tex", "w") as f:
            f.write(
                LATEX_TABLE_TEMPLATE.replace(
                    "{input_file}", f"paper_results_{metric}.tex"
                ).replace("{metric}", metric)
            )

        subprocess.run(f"lualatex render.tex", shell=True, check=True, cwd=tmpdir)
        shutil.copy2(
            tmpdir / "render.pdf",
            out_dir / f"paper_results_{metric}.pdf",
        )
