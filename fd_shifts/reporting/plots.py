from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fd_shifts.reporting import tables


def make_rank_df(dff: pd.DataFrame):
    select_df = dff  # [~dff.confid.str.startswith("VIT")]
    rank_df = select_df.rank(na_option="keep", numeric_only=True, ascending=False)

    confid = dff.index.get_level_values(0)
    classifier = dff.index.get_level_values(1)

    rank_df["confid"] = (
        classifier.where(classifier == "ViT", "").where(classifier != "ViT", "VIT-")
        + confid
    )

    return rank_df


def make_color_dict(rank_df: pd.DataFrame):
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


def plot_rank_style(data: pd.DataFrame, exp: str, metric: str, out_dir: Path):
    # dff = _aggregate_over_runs(data)
    dff = tables.aggregate_over_runs(data)
    dff = tables.build_results_table(dff, metric)
    rank_df = make_rank_df(dff)
    color_dict = make_color_dict(rank_df)

    def fix_studies(n):
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
        "sub-class-shift": "sub",  # sub
        "corruption-shift-1": "cor1",  # cor
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
        "failauc": "AUROC",
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

    studies0 = [
        "iid-study",  # iid
    ]
    studies1 = [
        #         "sub-class-shift",  # sub
        "corruption-shift-1",  # cor
        "corruption-shift-2",
        "corruption-shift-3",
        "corruption-shift-4",
        "corruption-shift-5",
    ]
    studies2 = [
        #         "new-class-shift-cifar10",  # s-ncs (includes openset)
        #         # 'new-class-shift-cifar10-original-mode',
        #         "new-class-shift-cifar100",  # ns-ncs
        #         # 'new-class-shift-cifar100-original-mode',
        #         "new-class-shift-svhn",
        #         # 'new-class-shift-svhn-original-mode',
        #         "new-class-shift-tinyimagenet",
        #         # 'new-class-shift-tinyimagenet-original-mode'
    ]
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
            "MCD-PE",
            "MSR",
            "PE",
            "MAHA",
            "DG-Res",
            "DG-MCD-EE",
        ]
        #     print(data.confid)
        plot_data = plot_data[plot_data.confid.str.replace("VIT-", "").isin(confids)]
        # plot_data = plot_data[~(plot_data.confid.isin(["VIT-Devries et al.", "VIT-ConfidNet", "VIT-DG-Res", "VIT-DG-Res"]))]
        plot_data = plot_data[plot_data.confid.str.startswith("VIT")]
        # print(plot_data.confid)
        plot_data.confid = plot_data.confid.str.replace("VIT-", "")
        plot_data["study"] = plot_data.study.apply(fix_studies)
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
        #         twin1 = axs.twinx()
        #         twin1.yaxis.tick_left()
        #         twin1.spines["left"].set_position(("data", x2[0]))
        for c in plot_data0["confid"]:
            confid_data0 = plot_data0[plot_data0["confid"] == c][
                ["study", metric]
            ].reset_index()
            confid_data1 = plot_data1[plot_data1["confid"] == c][
                ["study", metric]
            ].reset_index()
            #             confid_data2 = plot_data2[plot_data2["confid"] == c][
            #                 ["study", metric]
            #             ].reset_index()
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
            #             twin1.plot(
            #                 x2,
            #                 confid_data2[metric],
            #                 color=color_dict[c],
            #                 marker="o",
            #                 linewidth=3.1,
            #                 ms=18,
            #             )
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
            #             patch = patches.ConnectionPatch(
            #                 xyA=(x1[-1], confid_data1[metric].iloc[-1]),
            #                 xyB=(x2[0], confid_data2[metric].iloc[0]),
            #                 coordsA="data",
            #                 coordsB="data",
            #                 axesA=twin0,
            #                 axesB=twin1,
            #                 arrowstyle="-",
            #                 linestyle=":",
            #                 linewidth=3.1,
            #                 color=color_dict[c],
            #             )
            #             twin1.add_artist(patch)
            # logger.info((-0.2, ranked_confids.index(c)/(len(ranked_confids) - 1)))
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
                # bbox=dict(facecolor=color_dict[c], edgecolor='None', alpha=0.5 ),
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
        #     axs.set_xticklabels(studies, rotation=90)
        axs.set_xticklabels([studies_dict[s] for s in studies], fontsize=fontsize)
        axs.set_xlim(0, x1[-1] + 0.5)

        ylim0 = [plot_data0[metric].min(), plot_data0[metric].max()]

        ylim0[0] -= 0.07 * (ylim0[1] - ylim0[0])
        ylim0[1] += 0.07 * (ylim0[1] - ylim0[0])
        # print(ylim0)
        axs.set_ylim(ylim0)
        axs.set_ylabel(metric_dict[metric], fontsize=1.6 * fontsize)
        axs.yaxis.set_label_position("right")
        axs.set_axisbelow(False)
        axs.grid(True)
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
        twin0.grid(True)
        for label in twin0.get_xticklabels() + twin0.get_yticklabels():
            label.set_fontsize(fontsize)
            label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.75))

    #         ylim2 = [plot_data2[metric].min(), plot_data2[metric].max()]
    #         ylim2[0] -= 0.07 * (ylim2[1] - ylim2[0])
    #         ylim2[1] += 0.07 * (ylim2[1] - ylim2[0])
    #         twin1.set_ylim(ylim2)
    #         twin1.set_axisbelow(False)
    #         twin1.spines["left"].set_linewidth(4)
    #         twin1.spines["left"].set_color("k")
    #         twin1.spines["left"].set_zorder(0.5)
    #         twin1.spines["right"].set_linewidth(4)
    #         twin1.spines["right"].set_color("k")
    #         twin1.spines["right"].set_zorder(0.5)
    #         twin1.grid(False)
    #         for label in twin1.get_xticklabels() + twin1.get_yticklabels():
    #             label.set_fontsize(18)
    #             label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.75))

    plt.tight_layout()
    plt.savefig(out_dir / f"main_plot.png")
    # plt.show()
    # plt.close(f)


def vit_v_cnn_box(data: pd.DataFrame, out_dir: Path):
    dff = tables.aggregate_over_runs(data)
    dff = tables.build_results_table(dff, "aurc")
    rank_df = make_rank_df(dff)
    color_dict = make_color_dict(rank_df)
    print(data.columns)

    meanprops = dict(linestyle="-", linewidth=6, color="k", alpha=1, zorder=99)
    whiskerprops = dict(linestyle="-", linewidth=0)

    plot_exps = [
        "cifar10",
        "cifar100",
        "svhn",
        "breeds",
        "animals",
        "camelyon",
    ]  # exp_names
    cross_mode = False
    scale = 15
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=scale * 0.35)
    dim = "confid"

    metric_dict = {
        "aurc": "AURC",
        "failauc": "AUROC",
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

    studies = [
        "iid-study",
        # 'sub-class-shift',
        # 'corruption-shift-1',
        # 'corruption-shift-2',
        # 'corruption-shift-3',
        # 'corruption-shift-4',
        # 'corruption-shift-5',
        # 'new-class-shift-cifar10',
        # 'new-class-shift-cifar10-original-mode',
        # 'new-class-shift-cifar100',
        # 'new-class-shift-cifar100-original-mode',
        # 'new-class-shift-svhn',
        # 'new-class-shift-svhn-original-mode',
        # 'new-class-shift-tinyimagenet',
        # 'new-class-shift-tinyimagenet-original-mode'
    ]

    # print(df)

    def fix_studies(n):
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
        ]  # & (~df.confid.str.contains("DG"))]

        y = metric
        tmp_data = plot_data.assign(
            dataset=lambda row: row.study.str.split("_", 1, expand=True)[0]
        )
        tmp_data["backbone"] = tmp_data[dim]
        tmp_data.loc[~tmp_data[dim].str.startswith("VIT"), "backbone"] = "CNN"
        tmp_data.loc[tmp_data[dim].str.startswith("VIT"), "backbone"] = "VIT"
        confids = [
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
        #     print(data.confid)
        tmp_data = tmp_data[plot_data.confid.str.replace("VIT-", "").isin(confids)]
        tmp_data = tmp_data[
            ~(
                tmp_data.confid.isin(
                    ["VIT-Devries et al.", "VIT-ConfidNet", "VIT-DG-Res", "VIT-DG-Res"]
                )
            )
        ]
        #     logger.info(data.dataset)
        plot_colors = [color_dict[conf] for conf in tmp_data.confid.unique().tolist()]
        # print(plot_colors)
        palette = sns.color_palette(plot_colors)
        # print(plot_colors)
        # print(data.confid.unique().tolist())
        sns.set_palette(palette)

        # print(data[~data[dim].str.startswith("VIT")])

        # order = data[dim].str.replace("VIT-", "").sort_values().unique()

        # if not "noise" in study or "noise_study_3" in study:
        # print(study)
        # sns.stripplot(
        #     ax=saxs[yix],
        #     x=data[~data[dim].str.startswith("VIT")][dim],
        #     y=metric,
        #     data=data[~data[dim].str.startswith("VIT")],
        #     s=scale * 1.6,
        #     label=dim,
        #     order=order,
        # )
        # sns.stripplot(
        #     ax=saxs[yix],
        #     x=data[data[dim].str.startswith("VIT")][dim].str.replace("VIT-", ""),
        #     y=metric,
        #     data=data[data[dim].str.startswith("VIT")],
        #     s=scale * 1.6,
        #     label=dim,
        #     marker='X',
        #     order=order,
        # )
        for i, exp in enumerate(plot_exps):
            axs_ = axs.twinx()
            axs_.yaxis.tick_left()
            axs_.spines["left"].set_position(("data", i - 0.5))
            sns.boxplot(
                ax=axs_,
                x="dataset",
                y=metric,
                hue="backbone",
                data=tmp_data[tmp_data.dataset == exp],
                # medianprops=dict(alpha=0),
                medianprops=meanprops,
                saturation=1,
                showbox=True,
                showcaps=False,
                showfliers=False,
                whiskerprops=whiskerprops,
                showmeans=True,
                meanprops=dict(alpha=0),
                # meanprops=meanprops,
                # meanline=True,
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
        # axs[yix].set_xticklabels("")
        axs.set_xticklabels([dataset_dict[exp] for exp in plot_exps])

        # axs.set_title(fix_studies(study), pad=35)
        axs.set_ylabel(metric_dict[metric])
        axs.set_xlabel("")
        axs.yaxis.set_visible(False)
        axs.grid(False)

    # lim0 = data[metric].mean() - data[metric].std()
    # lim1 = data[metric].mean() + data[metric].std()
    # saxs[yix].set_ylim(lim0, lim1)
    # if yix == 0:
    #     saxs[yix].set_ylabel(metric)

    # if yix == 5:
    #     axs[yix].axis("off")
    #     axs[yix-1].legend()

    # if "iid" in study and metric == "aurc":
    #     axs[xix, yix].set_ylim(4, 8)
    # if "iid" in study and metric == "failauc":
    #     axs[xix, yix].set_ylim(0.90, 0.96)
    plt.tight_layout()
    # plt.savefig(
    #     "/home/tillb/Projects/failure-detection-benchmark/results/final_paper_{}_single_column_box.png".format(
    #         exp
    #     )
    # )
    plt.savefig(out_dir / f"vit_v_cnn.png")
