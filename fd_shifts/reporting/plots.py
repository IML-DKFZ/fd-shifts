from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import transforms

from fd_shifts.reporting import tables


def _make_rank_df(dff: pd.DataFrame):
    select_df = dff
    rank_df = select_df.rank(na_option="keep", numeric_only=True, ascending=False)

    confid = dff.index.get_level_values(0)
    classifier = dff.index.get_level_values(1)

    rank_df["confid"] = (
        classifier.where(classifier == "ViT", "").where(classifier != "ViT", "VIT-")
        + confid
    )

    return rank_df


def _make_color_dict(rank_df: pd.DataFrame):
    colors = [
        "tab:blue",
        "green",
        "tab:purple",
        "orange",
        "red",
        "black",
        "pink",
        "olive",
        "grey",
        "brown",
        "tab:cyan",
        "blue",
        "limegreen",
        "darkmagenta",
        "salmon",
        "tab:blue",
        "green",
        "tab:purple",
        "orange",
    ]

    color_dict = {
        conf: colors[ix]
        for ix, conf in enumerate(
            sorted(rank_df.confid.str.replace("VIT-", "").unique().tolist())
        )
    }
    color_dict.update(
        {
            conf: color_dict[conf.replace("VIT-", "")]
            for ix, conf in enumerate(
                rank_df.confid[rank_df.confid.str.contains("VIT")].tolist()
            )
        }
    )
    return color_dict


def plot_rank_style(data: pd.DataFrame, exp: str, metric: str, out_dir: Path) -> None:
    """Plot confid results over shifts

    Args:
        data (pd.DataFrame): cleaned experiment data
        exp (str): experiment (dataset) to consider
        metric (str): metric to consider
        out_dir (Path): where to save the created figure to
    """
    dff = tables.aggregate_over_runs(data)
    dff = tables.build_results_table(dff, metric)
    rank_df = _make_rank_df(dff)
    color_dict = _make_color_dict(rank_df)

    def _fix_studies(n):
        n = n.replace(exp + "_", "")
        n = n.replace("_proposed_mode", "")
        n = n.replace("_", "-")
        n = n.replace("-study-", "-shift-")
        n = n.replace("in-class", "sub-class")
        n = n.replace("noise", "corruption")
        n = n.replace("-resize", "")
        n = n.replace("-wilds-ood-test", "")
        n = n.replace("-ood-test", "")
        n = n.replace("-superclasses", "")
        return n

    studies_dict = {
        "iid-study": "iid",
        "sub-class-shift": "sub",
        "corruption-shift-1": "cor1",
        "corruption-shift-2": "cor2",
        "corruption-shift-3": "cor3",
        "corruption-shift-4": "cor4",
        "corruption-shift-5": "cor5",
        "new-class-shift-cifar10": "s-ncs\nc10",  # s-ncs (includes openset)
        "new-class-shift-cifar100": "s-ncs\nc100",  # s-ncs (includes openset)
        "new-class-shift-svhn": "ns-ncs\nsvhn",
        "new-class-shift-tinyimagenet": "ns-ncs\nti",
    }

    metric_dict = {
        "aurc": "AURC",
        "failauc": "$\\mathrm{AUROC}_f$",
        "ece": "ECE",
        "accuracy": "accuracy",
    }

    dataset_dict = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "animals": "iWildCam",
        "breeds": "BREEDS",
        "camelyon": "CAMELYON",
        "svhn": "SVHN",
    }

    scale = 10
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(6 * scale, 2.0 * scale * 1.2))
    fontsize = 48

    for axs, exp in zip(axes, ["cifar10", "cifar100"]):
        plot_data = data[data.study.str.startswith(exp + "_")][
            ["study", "confid", "run", metric]
        ]
        plot_data = plot_data[plot_data.confid != "VIT-ConfidNet-MCD"]
        confids = [
            "ConfidNet",
            "Devries et al.",
            "MCD-EE",
            "MCD-MSR",
            "MCD-MLS",
            "MCD-PE",
            "MCD-MI",
            "MSR",
            "MLS",
            "PE",
            "MAHA",
            "DG-Res",
            "DG-MCD-MSR",
        ]
        plot_data = plot_data[plot_data.confid.str.replace("VIT-", "").isin(confids)]
        plot_data = plot_data[plot_data.confid.str.startswith("VIT")]
        plot_data.confid = plot_data.confid.str.replace("VIT-", "")
        plot_data["study"] = plot_data.study.apply(_fix_studies)
        plot_data = plot_data[plot_data.study.isin(studies0 + studies1 + studies2)]
        plot_data0 = plot_data[plot_data.study.isin(studies0)]
        plot_data0 = plot_data0.groupby(["confid", "study"]).mean().reset_index()
        plot_data0 = plot_data0.sort_values(
            by="study", key=lambda x: x.apply(studies0.index)
        )
        plot_data1 = plot_data[plot_data.study.isin(studies1)]
        plot_data1 = plot_data1.groupby(["confid", "study"]).mean().reset_index()
        plot_data1 = plot_data1.sort_values(
            by="study", key=lambda x: x.apply(studies1.index)
        )
        plot_data2 = plot_data[plot_data.study.isin(studies2)]
        plot_data2 = plot_data2.groupby(["confid", "study"]).mean().reset_index()
        plot_data2 = plot_data2.sort_values(
            by="study", key=lambda x: x.apply(studies2.index)
        )
        scale = 10
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=scale * 0.50)

        x0 = np.arange(len(plot_data0.study.unique()))
        x1 = np.arange(
            x0[-1] + 1,
            x0[-1] + 1 + len(plot_data1.study.unique()) / 2,
            0.5,
        )
        x2 = np.arange(
            x1[-1] + 1,
            x1[-1] + 1 + len(plot_data2.study.unique()),
        )
        ranked_confids = (
            plot_data0[plot_data0.study == studies0[0]]
            .sort_values(by=metric, ascending=True)
            .confid.to_list()
        )
        twin0 = axs.twinx()
        twin0.yaxis.tick_left()
        twin0.spines["left"].set_position(("data", x1[0]))
        for c in plot_data0["confid"]:
            confid_data0 = plot_data0[plot_data0["confid"] == c][
                ["study", metric]
            ].reset_index()
            confid_data1 = plot_data1[plot_data1["confid"] == c][
                ["study", metric]
            ].reset_index()
            axs.plot(
                x0,
                confid_data0[metric],
                color=color_dict[c],
                marker="o",
                linewidth=3.1,
                ms=18,
            )
            twin0.plot(
                x1,
                confid_data1[metric],
                color=color_dict[c],
                marker="o",
                linewidth=3.1,
                ms=18,
            )
            patch = patches.ConnectionPatch(
                xyA=(x0[-1], confid_data0[metric].iloc[-1]),
                xyB=(x1[0], confid_data1[metric].iloc[0]),
                coordsA="data",
                coordsB="data",
                axesA=axs,
                axesB=twin0,
                arrowstyle="-",
                linestyle=":",
                linewidth=3.1,
                color=color_dict[c],
            )
            twin0.add_artist(patch)
            axs.annotate(
                xy=(0, confid_data0[metric].iloc[0]),
                xytext=(-0.1, ranked_confids.index(c) / (len(ranked_confids) - 1)),
                textcoords="axes fraction",
                text=c,
                fontsize=fontsize,
                horizontalalignment="right",
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=3.1,
                    color=color_dict[c],
                    alpha=1,
                    relpos=(1, 0.5),
                ),
                zorder=1,
                bbox=dict(
                    facecolor="None", edgecolor=color_dict[c], alpha=1, linewidth=3.1
                ),
            )

        studies = (
            list(plot_data0.study.unique())
            + list(plot_data1.study.unique())
            + list(plot_data2.study.unique())
        )
        axs.set_xticks(np.concatenate((x0, x1, x2)))
        axs.set_xticklabels([studies_dict[s] for s in studies], fontsize=fontsize)
        axs.set_xlim(0, x1[-1] + 0.5)

        ylim0 = [plot_data0[metric].min(), plot_data0[metric].max()]

        ylim0[0] -= 0.07 * (ylim0[1] - ylim0[0])
        ylim0[1] += 0.07 * (ylim0[1] - ylim0[0])
        axs.set_ylim(ylim0)
        axs.set_ylabel(metric_dict[metric], fontsize=1.6 * fontsize)
        axs.yaxis.set_label_position("right")
        axs.set_axisbelow(False)
        axs.grid(False)
        axs.tick_params(axis="y")
        axs.spines["top"].set_linewidth(0)
        axs.spines["top"].set_zorder(0.5)
        axs.spines["bottom"].set_linewidth(4)
        axs.spines["bottom"].set_zorder(0.5)
        axs.spines["left"].set_linewidth(4)
        axs.spines["left"].set_color("k")
        axs.spines["left"].set_zorder(0.1)
        axs.spines["right"].set_linewidth(0)
        axs.spines["right"].set_color("k")
        axs.spines["right"].set_zorder(0.5)
        axs.set_title(dataset_dict[exp], fontsize=1.6 * fontsize)
        for label in axs.get_xticklabels() + axs.get_yticklabels():
            label.set_fontsize(fontsize)
            label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.75))

        ylim1 = [plot_data1[metric].min(), plot_data1[metric].max()]
        ylim1[0] -= 0.07 * (ylim1[1] - ylim1[0])
        ylim1[1] += 0.07 * (ylim1[1] - ylim1[0])
        twin0.set_ylim(ylim1)
        twin0.set_axisbelow(False)
        twin0.spines["top"].set_linewidth(0)
        twin0.spines["left"].set_linewidth(4)
        twin0.spines["left"].set_color("k")
        twin0.spines["left"].set_zorder(0.5)
        twin0.spines["right"].set_linewidth(0)
        twin0.spines["right"].set_color("k")
        twin0.spines["right"].set_zorder(0.5)
        twin0.grid(False)
        for label in twin0.get_xticklabels() + twin0.get_yticklabels():
            label.set_fontsize(fontsize)
            label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.75))

    plt.tight_layout()
    plt.savefig(out_dir / f"main_plot.png")


def vit_v_cnn_box(data: pd.DataFrame, out_dir: Path) -> None:
    """Create plots that compare ViT to CNN results

    Args:
        data (pd.DataFrame): cleaned experiment data
        out_dir (Path): where to save the figures to
    """
    dff = tables.aggregate_over_runs(data)
    dff = tables.build_results_table(dff, "aurc")
    rank_df = _make_rank_df(dff)
    color_dict = _make_color_dict(rank_df)

    meanprops = dict(linestyle="-", linewidth=6, color="k", alpha=1, zorder=99)
    whiskerprops = dict(linestyle="-", linewidth=0)

    plot_exps = [
        "animals",
        "breeds",
        "camelyon",
        "cifar100",
        "cifar10",
        "svhn",
    ]
    cross_mode = False
    scale = 15
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=scale * 0.35)
    dim = "confid"

    metric_dict = {
        "aurc": "AURC",
        "failauc": "$\\mathrm{AUROC}_f$",
        "ece": "ECE",
        "accuracy": "Accuracy",
    }

    dataset_dict = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "animals": "iWildCam",
        "breeds": "BREEDS",
        "camelyon": "CAMELYON",
        "svhn": "SVHN",
    }

    studies = [
        "iid-study",
    ]

    def _fix_studies(n):
        n = n.replace("^.*?_", "")
        n = n.replace("_proposed_mode", "")
        n = n.replace("_", "-")
        n = n.replace("-study-", "-shift-")
        n = n.replace("in-class", "sub-class")
        n = n.replace("noise", "corruption")
        n = n.replace("-resize", "")
        n = n.replace("-wilds-ood-test", "")
        n = n.replace("-ood-test", "")
        n = n.replace("-superclasses", "")
        return n

    f, axes = plt.subplots(
        nrows=4, ncols=1, figsize=(4 * scale * 1.2, 4 * scale * 1.2), squeeze=True
    )
    for axs, metric in zip(axes, ["aurc", "ece", "failauc", "accuracy"]):
        plot_data = data[["study", "confid", "run", metric]][
            (data.study.str.contains("iid"))
        ]

        y = metric
        tmp_data = plot_data.assign(
            dataset=lambda row: row.study.str.split("_", 1, expand=True)[0]
        )
        tmp_data["backbone"] = tmp_data[dim]
        tmp_data.loc[~tmp_data[dim].str.startswith("VIT"), "backbone"] = "CNN"
        tmp_data.loc[tmp_data[dim].str.startswith("VIT"), "backbone"] = "VIT"
        confids = [
            "ConfidNet",
            "DG-MCD-MSR",
            "DG-Res",
            "Devries et al.",
            "MCD-EE",
            "MCD-MSR",
            "MCD-PE",
            "MCD-MI",
            "MSR",
            "PE",
            "MAHA",
        ]
        tmp_data = tmp_data[plot_data.confid.str.replace("VIT-", "").isin(confids)]
        tmp_data = tmp_data[
            ~(
                tmp_data.confid.isin(
                    ["VIT-Devries et al.", "VIT-ConfidNet", "VIT-DG-Res", "VIT-DG-Res"]
                )
            )
        ]
        plot_colors = [color_dict[conf] for conf in tmp_data.confid.unique().tolist()]
        palette = sns.color_palette(plot_colors)
        sns.set_palette(palette)

        for i, exp in enumerate(plot_exps):
            axs_ = axs.twinx()
            axs_.yaxis.tick_left()
            axs_.spines["left"].set_position(("data", i - 0.5))
            sns.boxplot(
                ax=axs_,
                x="dataset",
                y=metric,
                hue="backbone",
                data=tmp_data[tmp_data.dataset == exp].sort_values("backbone"),
                medianprops=meanprops,
                saturation=1,
                showbox=True,
                showcaps=False,
                showfliers=False,
                whiskerprops=whiskerprops,
                showmeans=True,
                meanprops=dict(alpha=0),
                order=plot_exps,
                boxprops=dict(alpha=0.5),
            )

            for label in axs_.get_xticklabels() + axs_.get_yticklabels():
                label.set_fontsize(28)
                label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.75))

            axs_.grid(False)
            if i != (len(plot_exps) - 1) or metric != "aurc":
                axs_.legend().remove()
            else:
                handles, labels = axs_.get_legend_handles_labels()
                axs_.legend(handles, ["CNN", "ViT"], title="classifier")

            if i != 0:
                axs_.set_ylabel("")
            else:
                axs_.yaxis.set_label_position("left")
                axs_.set_ylabel(metric_dict[metric])
        axs.set_xticklabels([dataset_dict[exp] for exp in plot_exps])

        axs.set_ylabel(metric_dict[metric])
        axs.set_xlabel("")
        axs.yaxis.set_visible(False)
        axs.grid(False)

    plt.tight_layout()
    plt.savefig(out_dir / f"vit_v_cnn.png")
