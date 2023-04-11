import shutil
import subprocess
import tempfile
from itertools import product
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
\usepackage{array}

\begin{document}

\begin{table*}%[h!]
\centering
\caption{\textbf{FD-Shifts Benchmark Results measured as {metric}}}
\label{tab:results}
\vspace{0.3cm}
\scalebox{0.265}{
\newcolumntype{?}{!{\vrule width 2pt}}
\newcolumntype{h}{!{\textcolor{white}{\vrule width 36pt}}}
\newcolumntype{x}{w{r}{2.5em}}
\renewcommand{\arraystretch}{1.5}
\huge
\input{{input_file}}}
\end{table*}
\end{document}
"""

LATEX_TABLE_TEMPLATE_LANDSCAPE = r"""
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
\usepackage{array}
\usepackage{pdflscape}

\begin{document}

\begin{landscape}
\begin{table*}%[h!]
\centering
\caption{\textbf{FD-Shifts Benchmark Results measured as {metric}}}
\label{tab:results}
\vspace{0.3cm}
\scalebox{0.165}{
\newcolumntype{?}{!{\vrule width 2pt}}
\newcolumntype{h}{!{\textcolor{white}{\vrule width 36pt}}}
\newcolumntype{x}{w{r}{2.5em}}
\renewcommand{\arraystretch}{1.8}
\huge
\input{{input_file}}}
\end{table*}
\end{landscape}
\end{document}
"""


def aggregate_over_runs(data: pd.DataFrame) -> pd.DataFrame:
    """Compute means over equivalent runs

    Args:
        data (pd.DataFrame): experiment data

    Returns:
        aggregated experiment data
    """
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


def _reorder_studies(
    table: pd.DataFrame, add_level: list[str] | None = None
) -> pd.DataFrame:
    """Reorder studies that are in a well-defind order for publication, append others to the end"""
    ordered_columns = [
        ("animals", "iid", ""),
        ("animals", "sub", ""),
        ("animals", "s-ncs", ""),
        ("animals", "rank", ""),
        ("breeds", "iid", ""),
        ("breeds", "sub", ""),
        ("breeds", "rank", ""),
        ("camelyon", "iid", ""),
        ("camelyon", "sub", ""),
        ("camelyon", "rank", ""),
        ("cifar100", "iid", ""),
        ("cifar100", "sub", ""),
        ("cifar100", "cor", ""),
        ("cifar100", "s-ncs", "c10"),
        ("cifar100", "ns-ncs", "svhn"),
        ("cifar100", "ns-ncs", "ti"),
        ("cifar100", "rank", ""),
        ("cifar10", "iid", ""),
        ("cifar10", "cor", ""),
        ("cifar10", "s-ncs", "c100"),
        ("cifar10", "ns-ncs", "svhn"),
        ("cifar10", "ns-ncs", "ti"),
        ("cifar10", "rank", ""),
        ("svhn", "iid", ""),
        ("svhn", "s-ncs", ""),
        ("svhn", "ns-ncs", "c10"),
        ("svhn", "ns-ncs", "c100"),
        ("svhn", "ns-ncs", "ti"),
        ("svhn", "rank", ""),
    ]

    if add_level is not None:
        ordered_columns = list(
            map(lambda t: t[0] + (t[1],), product(ordered_columns, add_level))
        )

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


def _build_multilabel(table: pd.DataFrame, paper_filter: bool = True) -> pd.DataFrame:
    table.columns = table.columns.get_level_values(0).map(_study_name_to_multilabel)
    table.confid = table.confid.str.replace("VIT-", "")

    if paper_filter:
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
    return data


def build_results_table(
    data: pd.DataFrame,
    metric: str,
    original_mode: bool = False,
    paper_filter: bool = True,
) -> pd.DataFrame:
    """Create results table from aggregated and cleaned experiment data

    Args:
        data (pd.DataFrame): experiment data
        metric (str): metric to report
        original_mode (bool): whether to report proposed or original handling of id
            misclassifications
        paper_filter (bool): whether to apply filters that are specific to the publication

    Returns:
        results table
    """
    results_table = _create_results_pivot(data, metric, original_mode)
    if paper_filter:
        results_table = _aggregate_noise_studies(results_table, metric)

    results_table = _build_multilabel(results_table, paper_filter)

    results_table = _reorder_studies(results_table)
    if paper_filter:
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


def _add_rank_columns(data: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    print("_add_rank_columns")
    print(data.columns)

    _rank_table = data.mask(
        data.apply(
            lambda row: row.index.get_level_values(1).str.contains("ViT"), axis=0
        ),
        data.applymap(float)
        .loc[
            data.index[data.index.get_level_values(1).str.contains("ViT")],
            data.columns,
        ]
        .rank(axis=0, ascending=ascending, numeric_only=True, method="min"),
    ).mask(
        data.apply(
            lambda row: ~row.index.get_level_values(1).str.contains("ViT"), axis=0
        ),
        data.applymap(float)
        .loc[
            data.index[~data.index.get_level_values(1).str.contains("ViT")],
            data.columns,
        ]
        .rank(axis=0, ascending=ascending, numeric_only=True, method="min"),
    )

    return _rank_table


def paper_results(
    data: pd.DataFrame,
    metric: str,
    invert: bool,
    out_dir: Path,
    rank_cols: bool = False,
):
    """Create colored results table in tex format

    Args:
        data (pd.DataFrame): cleaned up experiment data
        metric (str): metric to report
        invert (bool): whether for this metric lower is better or not
        out_dir (Path): where to save the output to
        rank_cols: (bool): whether to report ranks instead of absolute values
    """
    _formatter = (
        lambda x: f"{x:>3.2f}"[:4] if "." in f"{x:>3.2f}"[:3] else f"{x:>3.2f}"[:3]
    )
    results_table = build_results_table(data, metric)
    cmap = "Oranges_r" if invert else "Oranges"

    if rank_cols:
        results_table = _add_rank_columns(results_table)
        print(f"{metric}")
        print(results_table)
        _formatter = lambda x: f"{int(x):>3d}"
        cmap = "Oranges"

    results_table = results_table.rename(
        columns=_dataset_to_display_name,
        level=0,
    )

    sanity_check(results_table, metric)

    # Render table
    results_table = results_table.astype(float).applymap(
        lambda val: round(val, 2) if val < 10 else round(val, 1)
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
            _formatter,
            na_rep="*",
        )
    )

    ltex.data.columns = ltex.data.columns.set_names(
        ["\\multicolumn{1}{c}{}", "study", "ncs-data set"]
    )
    ltex = ltex.to_latex(
        convert_css=True,
        hrules=True,
        multicol_align="c?",
        column_format="ll?rrr?xx?xx?rrrrrr?rrrrr?rrrrr",
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

    with open(
        out_dir / f"paper_results_{metric}{'_rank' if rank_cols else ''}.tex", "w"
    ) as f:
        f.write(ltex)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy2(
            out_dir / f"paper_results_{metric}{'_rank' if rank_cols else ''}.tex",
            tmpdir / f"paper_results_{metric}{'_rank' if rank_cols else ''}.tex",
        )
        with open(tmpdir / "render.tex", "w") as f:
            f.write(
                LATEX_TABLE_TEMPLATE.replace(
                    "{input_file}",
                    f"paper_results_{metric}{'_rank' if rank_cols else ''}.tex",
                ).replace("{metric}", metric)
            )

        subprocess.run(f"lualatex render.tex", shell=True, check=True, cwd=tmpdir)
        shutil.copy2(
            tmpdir / "render.pdf",
            out_dir / f"paper_results_{metric}{'_rank' if rank_cols else ''}.pdf",
        )


def rank_comparison_metric(data: pd.DataFrame, out_dir: Path):
    """Create colored results table in tex format to compare ranking between metrics

    Args:
        data (pd.DataFrame): cleaned up experiment data
        out_dir (Path): where to save the output to
    """
    aurc_table = build_results_table(data, "aurc")
    aurc_table = _add_rank_columns(aurc_table)
    aurc_table.columns = pd.MultiIndex.from_tuples(
        map(lambda t: t + (r"$\alpha$",), aurc_table.columns)
    )

    failauc_table = build_results_table(data, "failauc")
    failauc_table = _add_rank_columns(failauc_table, False)
    failauc_table.columns = pd.MultiIndex.from_tuples(
        map(lambda t: t + (r"$\beta$",), failauc_table.columns)
    )

    results_table = pd.concat((aurc_table, failauc_table), axis=1)
    results_table = _reorder_studies(results_table, add_level=[r"$\alpha$", r"$\beta$"])

    _formatter = lambda x: f"{int(x):>3d}"

    results_table = results_table.rename(
        columns=_dataset_to_display_name,
        level=0,
    )

    # Render table
    cmap = "Oranges"
    results_table = results_table.astype(float).applymap(
        lambda val: round(val, 2) if val < 10 else round(val, 1)
    )

    gmap_vit = _compute_gmap(
        results_table.loc[
            results_table.index[
                results_table.index.get_level_values(1).str.contains("ViT")
            ],
            results_table.columns,
        ],
        True,
    )
    gmap_cnn = _compute_gmap(
        results_table.loc[
            results_table.index[
                ~results_table.index.get_level_values(1).str.contains("ViT")
            ],
            results_table.columns,
        ],
        True,
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
            _formatter,
            na_rep="*",
        )
    )

    ltex.data.columns = ltex.data.columns.set_names(
        ["\\multicolumn{1}{c}{}", "study", "ncs-data set", "metric"]
    )
    print(len(results_table.columns))
    ltex = ltex.to_latex(
        convert_css=True,
        hrules=True,
        multicol_align="c?",
        column_format=(
            "ll?"
            + 3 * "*{2}{r}h"
            + 2 * "*{2}{r}h"
            + 2 * "*{2}{r}h"
            + 6 * "*{2}{r}h"
            + 5 * "*{2}{r}h"
            + 4 * "*{2}{r}h"
            + "*{2}{r}"
        ),
    )

    # Remove toprule
    ltex = list(filter(lambda line: line != r"\toprule", ltex.splitlines()))

    # No separators in first header row

    # Remove last separator in second header row
    # (this is just `replace("?", "", 1)`, but from the right)
    ltex[1] = ltex[1][: ltex[1].rfind("?")] + ltex[1][ltex[1].rfind("?") + 1 :]
    ltex[2] = ltex[2][: ltex[2].rfind("?")] + ltex[2][ltex[2].rfind("?") + 1 :]
    ltex[3] = ltex[3][: ltex[3].rfind("?")] + ltex[3][ltex[3].rfind("?") + 1 :]

    # Insert empty row before ViT part
    i = ltex.index(next((x for x in ltex if "ViT" in x)))
    ltex.insert(i, "\\midrule \\\\")

    ltex = "\n".join(ltex)

    with open(out_dir / f"rank_metric_comparison.tex", "w") as f:
        f.write(ltex)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy2(
            out_dir / f"rank_metric_comparison.tex",
            tmpdir / f"rank_metric_comparison.tex",
        )
        with open(tmpdir / "render.tex", "w") as f:
            f.write(
                LATEX_TABLE_TEMPLATE_LANDSCAPE.replace(
                    "{input_file}", f"rank_metric_comparison.tex"
                ).replace("{metric}", "")
            )

        subprocess.run(f"lualatex render.tex", shell=True, check=True, cwd=tmpdir)
        shutil.copy2(
            tmpdir / "render.pdf",
            out_dir / f"rank_metric_comparison.pdf",
        )


def rank_comparison_mode(data: pd.DataFrame, out_dir: Path, rank: bool = True):
    """Create colored results table in tex format to compare ranking between handling of iid
    missclassifications

    Args:
        data (pd.DataFrame): cleaned up experiment data
        out_dir (Path): where to save the output to
        rank: (bool): whether to report ranks instead of absolute values
    """
    prop_table = build_results_table(data, "failauc", original_mode=False)
    if rank:
        prop_table = _add_rank_columns(prop_table, False)

    prop_table.columns = pd.MultiIndex.from_tuples(
        map(lambda t: t + ("P",), prop_table.columns)
    )

    orig_table = build_results_table(data, "failauc", original_mode=True)
    cmap = "Oranges_r"
    if rank:
        orig_table = _add_rank_columns(orig_table, False)
        cmap = "Oranges"
    orig_table.columns = pd.MultiIndex.from_tuples(
        map(lambda t: t + ("O",), orig_table.columns)
    )

    results_table = pd.concat((prop_table, orig_table), axis=1)
    results_table = results_table[
        list(filter(lambda t: "ncs" in t[1], results_table.columns))
    ]
    results_table = _reorder_studies(results_table, add_level=["P", "O"])

    if rank:
        _formatter = lambda x: f"{int(x):>3d}"
    else:
        _formatter = (
            lambda x: f"{x:>3.2f}"[:4] if "." in f"{x:>3.2f}"[:3] else f"{x:>3.2f}"[:3]
        )

    results_table = results_table.rename(
        columns=_dataset_to_display_name,
        level=0,
    )

    # Render table
    results_table = results_table.astype(float).applymap(
        lambda val: round(val, 2) if val < 10 else round(val, 1)
    )

    gmap_vit = _compute_gmap(
        results_table.loc[
            results_table.index[
                results_table.index.get_level_values(1).str.contains("ViT")
            ],
            results_table.columns,
        ],
        True,
    )
    gmap_cnn = _compute_gmap(
        results_table.loc[
            results_table.index[
                ~results_table.index.get_level_values(1).str.contains("ViT")
            ],
            results_table.columns,
        ],
        True,
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
            _formatter,
            na_rep="*",
        )
    )

    ltex.data.columns = ltex.data.columns.set_names(
        ["\\multicolumn{1}{c}{}", "study", "ncs-data set", "ood protocol"]
    )
    print(len(results_table.columns))
    ltex = ltex.to_latex(
        convert_css=True,
        hrules=True,
        multicol_align="c?",
        column_format=(
            "ll?"
            + 1 * "*{2}{r}h"
            + 3 * "*{2}{r}h"
            + 3 * "*{2}{r}h"
            + 3 * "*{2}{r}h"
            + "*{2}{r}"
        ),
    )

    # Remove toprule
    ltex = list(filter(lambda line: line != r"\toprule", ltex.splitlines()))

    # No separators in first header row

    # Remove last separator in second header row
    # (this is just `replace("?", "", 1)`, but from the right)
    ltex[1] = ltex[1][: ltex[1].rfind("?")] + ltex[1][ltex[1].rfind("?") + 1 :]
    ltex[2] = ltex[2][: ltex[2].rfind("?")] + ltex[2][ltex[2].rfind("?") + 1 :]
    ltex[3] = ltex[3][: ltex[3].rfind("?")] + ltex[3][ltex[3].rfind("?") + 1 :]

    # Insert empty row before ViT part
    i = ltex.index(next((x for x in ltex if "ViT" in x)))
    ltex.insert(i, "\\midrule \\\\")

    ltex = "\n".join(ltex)

    with open(out_dir / f"{'rank_' if rank else ''}mode_comparison.tex", "w") as f:
        f.write(ltex)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy2(
            out_dir / f"{'rank_' if rank else ''}mode_comparison.tex",
            tmpdir / f"{'rank_' if rank else ''}mode_comparison.tex",
        )
        with open(tmpdir / "render.tex", "w") as f:
            f.write(
                LATEX_TABLE_TEMPLATE.replace(
                    "{input_file}", f"{'rank_' if rank else ''}mode_comparison.tex"
                ).replace("{metric}", "")
            )

        subprocess.run(f"lualatex render.tex", shell=True, check=True, cwd=tmpdir)
        shutil.copy2(
            tmpdir / "render.pdf",
            out_dir / f"{'rank_' if rank else ''}mode_comparison.pdf",
        )
