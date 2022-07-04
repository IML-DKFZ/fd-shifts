from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

metric = "aurc"


def aggregate_over_runs(df):
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


def make_rank_df(dff):
    select_df = dff  # [~dff.confid.str.startswith("VIT")]
    rank_df = select_df.rank(na_option="keep", numeric_only=True, ascending=False)
    # actually aurc should be ranked ascedingly, but we want the lowest rank to show on top on the y axis
    # so careful when using this df for other things than this plot!

    rank_df["confid"] = dff.confid
    #     print(select_df)
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
    dff = aggregate_over_runs(data)
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
        print(ylim0)
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


def plot_sum_ranking(data: pd.DataFrame, out_dir: Path):
    dff = aggregate_over_runs(data)
    rank_df = make_rank_df(dff)
    color_dict = make_color_dict(rank_df)

    select_columns = [c for c in rank_df.columns]
    iid_columns = [c for c in select_columns if "iid" in c]
    print("IID", iid_columns)
    in_class_columns = [c for c in select_columns if "in_class" in c]
    print("SUB CLASS", in_class_columns)
    new_class_columns = [
        c for c in select_columns if ("new_class" in c and "proposed" in c)
    ]
    sem_new_class_columns = [
        c for c in new_class_columns if ("cifar10_" in c and "cifar100_" in c)
    ]
    print("SEMANTIC NEW CLASS", sem_new_class_columns)
    nonsem_new_class_columns = [
        c for c in new_class_columns if c not in sem_new_class_columns
    ]
    print("NON-SEMANTIC NEW CLASS", nonsem_new_class_columns)
    noise_columns = [c for c in select_columns if "noise" in c]
    print("NOISE", noise_columns)
    sum_rank_df = rank_df[["confid"]]
    sum_rank_df.loc[rank_df.confid.str.startswith("VIT"), "confid"] = "VIT"
    sum_rank_df.loc[~rank_df.confid.str.startswith("VIT"), "confid"] = "CNN"
    # logger.info(sum_rank_df)
    # print(rank_df[rank_df.isna()])
    skipna = False
    sum_rank_df["iid"] = rank_df[iid_columns].sum(
        axis=1, numeric_only=True, skipna=skipna
    )
    sum_rank_df["corruption-shift"] = rank_df[noise_columns].sum(
        axis=1, numeric_only=True, skipna=skipna
    )
    if len(in_class_columns) > 0:
        sum_rank_df["sub-class-shift"] = rank_df[in_class_columns].sum(
            axis=1, numeric_only=True, skipna=skipna
        )
    sum_rank_df["sem.-new-class-shift"] = rank_df[sem_new_class_columns].sum(
        axis=1, numeric_only=True, skipna=skipna
    )
    sum_rank_df["non-sem.-new-class-shift"] = rank_df[nonsem_new_class_columns].sum(
        axis=1, numeric_only=True, skipna=skipna
    )
    sum_rank_df = sum_rank_df.groupby("confid").sum()
    sum_rank_df = sum_rank_df.reset_index(drop=False)
    confids = sum_rank_df.confid
    sum_rank_df = sum_rank_df.rank(na_option="keep", numeric_only=True, ascending=True)
    sum_rank_df["confid"] = confids
    sum_rank_df["aggregated"] = sum_rank_df.sum(
        axis=1, numeric_only=True, skipna=skipna
    ).rank(na_option="keep", ascending=True)

    # sum_rank_df["iid"] = sum_rank_df.apply(lambda row: row["iid"] + 0.5 if row["confid"] == "confidnet_mcd" else row["iid"], axis=1)
    # sum_rank_df["iid"] = sum_rank_df.apply(lambda row: row["iid"] - 0.5 if row["confid"] == "deepgamblers_mcd_mi" else row["iid"], axis=1)

    scale = 10
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=scale * 0.50)
    f, axs = plt.subplots(nrows=1, ncols=1, figsize=(4 * scale, 1.5 * scale * 1.2))
    # todo ! supercifar has to be a part of cifar100 exp. check also weird observation regarding val_tuning

    show_columns = [
        "iid",
        "corruption-shift",
        "sub-class-shift",
        "sem.-new-class-shift",
        "non-sem.-new-class-shift",
        "aggregated",
    ]
    cols = show_columns  # [c for c in sum_rank_df.columns if c.startswith("sum")]
    numeric_exp_df = sum_rank_df[cols]
    # todo DROPNAN?
    confids_list = sum_rank_df.confid.tolist()
    x = range(len(numeric_exp_df.columns))
    ranked_confs = sum_rank_df.sort_values(by=numeric_exp_df.columns[0]).confid.tolist()
    # print(numeric_exp_df)
    # print(confids_list)
    import numpy as np

    seen = [{} for _ in x]
    for ix in range(len(numeric_exp_df)):
        y = numeric_exp_df.iloc[ix].values
        #     axs.plot(x, y, linewidth=3.1, marker=".", ms=18, color=color_dict[sum_rank_df.confid.tolist()[ix]])
        xprev = x[0]
        yprev = y[0]
        textprev = None
        for i, (x_, y_) in enumerate(zip(x, y)):
            if np.isnan(y_):
                continue

            if y_ in seen[x_].keys():
                text = seen[x_][y_]
                text.set_text(text.get_text() + "\n" + confids_list[ix])
            else:
                text = axs.text(
                    x_,
                    y_,
                    confids_list[ix],
                    fontsize=16,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                seen[x_][y_] = text

            if i == 0:
                arrowprops = None
                xycoords = "data"
                xy = (0, 0)
            else:
                arrowprops = dict(
                    arrowstyle="-",
                    linewidth=3.1,
                    color=list(color_dict.values())[ix],
                    relpos=(0, 0.5),
                    alpha=0.4,
                )
                xycoords = textprev
                xy = (1, 0.5)

            axs.annotate(
                text="",
                xy=xy,
                xytext=(0, 0.5),
                xycoords=xycoords,
                textcoords=text,
                fontsize=16,
                horizontalalignment="center",
                verticalalignment="center",
                arrowprops=arrowprops,
            )
            xprev = x_
            yprev = y_
            textprev = text
    #     break
    axs.set_yticks(range(1, len(numeric_exp_df) + 1))
    axs.set_yticks([])
    # axs.set_yticklabels(ranked_confs)
    axs.set_xticks(x)
    axs.set_xticklabels([c[:5] for c in numeric_exp_df.columns], rotation=90)
    axs.set_xlim(0, len(numeric_exp_df.columns) - 1)
    axs.set_ylim(0.5, 2.5)
    #     print(axs.get_facecolor())
    axs.annotate(
        "",
        xy=(1.05, 0),
        xytext=(1.05, 1),
        arrowprops=dict(width=3, headwidth=8, headlength=8, color="grey"),
        xycoords="axes fraction",
    )
    axs.annotate(
        "best\nrank",
        xy=(1.054, 1),
        xytext=(1.054, 1),
        xycoords="axes fraction",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="top",
    )
    axs.annotate(
        "worst\nrank",
        xy=(1.054, 0),
        xytext=(1.054, 0),
        xycoords="axes fraction",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    plt.tight_layout()
    # plt.savefig("/Users/Paul/research/files/analysis/paper_plots/ranking.png")
    # plt.savefig("/home/tillb/Projects/failure-detection-benchmark/results/ranking.png")
    plt.savefig(out_dir / "vit_v_cnn.png")
