from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import kendalltau, wilcoxon

plt.rcParams.update(
    {
        "font.size": 11.0,
        "font.family": "serif",
        "font.serif": "Palatino",
        "axes.titlesize": "medium",
        "figure.titlesize": "medium",
        "text.usetex": True,
    }
)


def _make_color_dict(confids):
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
    ]

    color_dict = {
        conf: colors[ix % len(colors)]
        for ix, conf in enumerate(sorted(list(np.unique(confids))))
    }
    return color_dict


def bs_box_scatter_plot(
    data: pd.DataFrame,
    metric: str,
    out_dir: Path,
    filename: str,
) -> None:
    """"""
    grouped_data = data.groupby("confid")[metric]

    # Create box plots for each confidence level
    plt.boxplot(
        [grouped_data.get_group(confid) for confid in grouped_data.groups.keys()],
        positions=np.arange(len(grouped_data)),
        patch_artist=True,
    )

    # Overlay scatter plot with jittered points
    for loc, (_, group) in enumerate(grouped_data):
        jitter = np.random.normal(loc=0, scale=0.05, size=len(group))
        plt.scatter([loc] * len(group) + jitter, group, alpha=0.5, color="k", s=2)

    plt.xticks(
        ticks=np.arange(len(grouped_data)),
        labels=grouped_data.groups.keys(),
        rotation=60,
        horizontalalignment="right",
        verticalalignment="top",
    )
    plt.ylabel(metric)
    plt.title(filename)
    plt.savefig(out_dir / filename, bbox_inches="tight")
    plt.close()


def bs_podium_plot(
    data: pd.DataFrame,
    metric: str,
    histograms: pd.DataFrame,
    out_dir: Path,
    filename: str,
) -> None:
    """"""
    n_confid = histograms.shape[0]
    rank_values = np.arange(1, n_confid + 1)
    # Assuming the histogram rows to be sorted by overall rank
    confid_to_rank_idx = {histograms.index[i]: i for i in range(n_confid)}

    # Create a color map for confid
    cmap = plt.cm.turbo
    color_list_confids = [cmap(r) for r in np.linspace(0, 1, n_confid)]
    colors_dict = _make_color_dict(histograms.index)

    _, (ax_scatter, ax_hist) = plt.subplots(
        nrows=2,
        ncols=1,
        height_ratios=(6, 1),
        gridspec_kw=dict(hspace=0),
        subplot_kw=dict(xlim=(0.8, n_confid + 1.2)),
    )

    plt.sca(ax_scatter)
    # Create scatter plot
    for confid, group in data.groupby("confid"):
        r = confid_to_rank_idx[confid]
        plt.scatter(
            group["rank"] + (r + 0.5) / n_confid,
            group[metric],
            label=confid,
            color=colors_dict[confid],
            s=3.5,
            # increase zorder such that the scatter plot is in front of the lines
            zorder=3,
        )

    # Create line plot connecting points with the same bootstrap_index
    for _, group in data.sort_values(by="rank").groupby(
        "bootstrap_index" if "run" not in data.columns else ["bootstrap_index", "run"]
    ):
        if len(group) != n_confid:
            raise ValueError(
                f"Missing results for the following group (expected {n_confid} "
                f"confids):\n\n{group}"
            )
        for i in range(n_confid - 1):
            plt.plot(
                [
                    group["rank"].values[i]
                    + (confid_to_rank_idx[group.confid.values[i]] + 0.5) / n_confid,
                    group["rank"].values[i + 1]
                    + (confid_to_rank_idx[group.confid.values[i + 1]] + 0.5) / n_confid,
                ],
                group[metric][i : i + 2],
                color=colors_dict[group.confid.values[i]],
                lw=0.08,
                # random zorder such that no color is completely hidden
                zorder=np.random.rand() + 1.5,
            )

    plt.vlines(
        np.arange(1, n_confid + 2),
        ymin=data[metric].min(),
        ymax=data[metric].max(),
        linestyles="dashed",
        colors="k",
        linewidths=0.5,
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=3)
    plt.ylabel(metric)
    plt.xticks(np.arange(1, n_confid + 2), labels=(n_confid + 1) * [""])
    plt.tick_params(axis="x", direction="in")

    # Histogram plots on lower axis
    plt.sca(ax_hist)
    for rank in histograms.columns:
        plt.bar(
            x=rank + np.linspace(0, 1, n_confid, endpoint=False),
            height=histograms[rank],
            width=1 / n_confid,
            align="edge",
            color=[colors_dict[confid] for confid in histograms.index],
        )
    plt.yticks(ticks=[])
    # Using hidden minor ticks to get centered labels on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.FixedLocator(np.arange(1, n_confid + 2)))
    plt.gca().xaxis.set_minor_locator(
        ticker.FixedLocator(np.arange(1, n_confid + 1) + 0.5)
    )
    plt.gca().xaxis.set_major_formatter(ticker.NullFormatter())
    plt.gca().xaxis.set_minor_formatter(ticker.FixedFormatter(list(rank_values) + [""]))
    plt.gca().tick_params(axis="x", which="minor", bottom=False)
    plt.xlabel("Rank")
    plt.title(filename)
    plt.savefig(out_dir / filename, bbox_inches="tight")
    plt.close()


def bs_blob_plot(
    histograms: pd.DataFrame,
    medians,
    out_dir: Path,
    filename: str,
) -> None:
    """"""
    max_blob_size = 300

    # Plot blobs
    n_samples = histograms.sum(axis=1)[0]
    n_confid = histograms.shape[0]
    rank_values = np.arange(1, n_confid + 1)

    # Reindex columns handling the case of shared last ranks
    histograms = histograms.reindex(columns=rank_values, fill_value=0)

    colors_dict = _make_color_dict(histograms.index)

    for idx, confid in enumerate(histograms.index):
        plt.scatter(
            n_confid * [idx],
            rank_values,
            s=histograms.loc[confid] / n_samples * max_blob_size,
            c=colors_dict[confid],
        )

    plt.plot(rank_values - 1, rank_values, color="gray", ls="dashed", lw=0.5, zorder=0)
    plt.scatter(
        rank_values - 1,
        medians,
        marker="x",
        color="k",
        s=0.8 * max_blob_size,
        linewidths=0.5,
    )

    plt.xticks(
        ticks=rank_values - 1,
        labels=list(histograms.index),
        rotation=60,
        horizontalalignment="right",
        verticalalignment="top",
    )
    plt.yticks(ticks=rank_values)
    plt.gca().set_axisbelow(True)
    plt.gca().xaxis.grid(color="gray", lw=1.5, alpha=0.15)
    plt.gca().yaxis.grid(color="gray", lw=1.5, alpha=0.15)
    plt.ylabel("Rank")
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()


def bs_significance_map(
    data: pd.DataFrame,
    metric: str,
    histograms: pd.DataFrame,
    out_dir: Path,
    filename: str,
) -> None:
    """"""
    # significance level
    alpha = 0.05
    n_confid = histograms.shape[0]
    rank_values = np.arange(1, n_confid + 1)
    confid_indices = np.arange(n_confid)

    # Reindex columns handling the case of shared last ranks
    histograms = histograms.reindex(columns=rank_values, fill_value=0)

    # Compute significance map
    significance = np.zeros((n_confid, n_confid))
    confid_names = histograms.index.values

    for i, j in product(range(n_confid), range(n_confid)):
        if i == j:
            significance[i, j] = np.nan
            continue

        # Catch the case where all values are the same and set significance to 0
        if np.allclose(
            data.groupby("confid")
            .get_group(confid_names[i])
            .sort_values(
                by=["bootstrap_index"]
                if "run" not in data.columns
                else ["bootstrap_index", "run"]
            )[metric],
            data.groupby("confid")
            .get_group(confid_names[j])
            .sort_values(
                by=["bootstrap_index"]
                if "run" not in data.columns
                else ["bootstrap_index", "run"]
            )[metric],
        ):
            significance[i, j] = 0
        else:
            # Get the two confid-groups and sort the values by bootstrap index and run to
            # ensure that they are aligned.
            significance[i, j] = int(
                wilcoxon(
                    data.groupby("confid")
                    .get_group(confid_names[i])
                    .sort_values(
                        by=["bootstrap_index"]
                        if "run" not in data.columns
                        else ["bootstrap_index", "run"]
                    )[metric],
                    data.groupby("confid")
                    .get_group(confid_names[j])
                    .sort_values(
                        by=["bootstrap_index"]
                        if "run" not in data.columns
                        else ["bootstrap_index", "run"]
                    )[metric],
                    correction=False,
                    alternative="less",
                ).pvalue
                < alpha
            )

    colors = ["steelblue", "yellow"]
    alphas = [0.85, 0.5]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()

    for i, c in enumerate(significance):
        for j, s in enumerate(c):
            if i == j:
                continue
            ax.add_patch(
                Rectangle(
                    (i - 0.5, j - 0.5),
                    width=1,
                    height=1,
                    color=colors[int(s)],
                    alpha=alphas[int(s)],
                    lw=0,
                    fill=True,
                    zorder=-2,
                )
            )

    plt.plot(
        [-0.6, n_confid - 0.4], [-0.6, n_confid - 0.4], color="k", lw=0.5, zorder=10
    )
    plt.grid(color="whitesmoke")

    plt.xticks(
        ticks=confid_indices + 0.4,
        labels=list(histograms.index),
        rotation=60,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=23,
        minor=True,
    )
    plt.yticks(
        ticks=confid_indices + 0.25,
        labels=list(histograms.index),
        rotation=30,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=23,
        minor=True,
    )

    plt.gca().tick_params(axis="x", which="minor", bottom=False)
    plt.xticks(ticks=confid_indices, labels=n_confid * [])
    plt.gca().tick_params(axis="y", which="minor", left=False)
    plt.yticks(ticks=confid_indices, labels=n_confid * [])

    plt.xlim(-0.6, n_confid - 0.4)
    plt.ylim(-0.6, n_confid - 0.4)
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()


def bs_significance_map_colored(
    data: pd.DataFrame,
    metric: str,
    histograms: pd.DataFrame,
    out_dir: Path,
    filename: str,
    no_labels: bool = False,
    flip_horizontally: bool = False,
) -> None:
    """"""
    # significance level
    alpha = 0.05
    n_confid = histograms.shape[0]
    rank_values = np.arange(1, n_confid + 1)
    confid_indices = np.arange(n_confid)

    # Reindex columns handling the case of shared last ranks
    histograms = histograms.reindex(columns=rank_values, fill_value=0)
    colors_dict = _make_color_dict(histograms.index)

    # Compute significance map
    significance = np.zeros((n_confid, n_confid))
    confid_names = histograms.index.values

    for i, j in product(range(n_confid), range(n_confid)):
        if i == j:
            significance[i, j] = np.nan
            continue

        # Catch the case where all values are the same and set significance to 0
        if np.allclose(
            data.groupby("confid")
            .get_group(confid_names[i])
            .sort_values(
                by=["bootstrap_index"]
                if "run" not in data.columns
                else ["bootstrap_index", "run"]
            )[metric],
            data.groupby("confid")
            .get_group(confid_names[j])
            .sort_values(
                by=["bootstrap_index"]
                if "run" not in data.columns
                else ["bootstrap_index", "run"]
            )[metric],
        ):
            significance[i, j] = 0
        else:
            # Get the two confid-groups and sort the values by bootstrap index and run to
            # ensure that they are aligned.
            significance[i, j] = int(
                wilcoxon(
                    data.groupby("confid")
                    .get_group(confid_names[i])
                    .sort_values(
                        by=["bootstrap_index"]
                        if "run" not in data.columns
                        else ["bootstrap_index", "run"]
                    )[metric],
                    data.groupby("confid")
                    .get_group(confid_names[j])
                    .sort_values(
                        by=["bootstrap_index"]
                        if "run" not in data.columns
                        else ["bootstrap_index", "run"]
                    )[metric],
                    correction=False,
                    alternative="less",
                ).pvalue
                < alpha
            )

    if no_labels:
        # Create legend separately
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        for i, (confid, color) in enumerate(colors_dict.items()):
            ax.add_patch(
                Rectangle(
                    (0, 2 * i),
                    width=1,
                    height=1,
                    color=colors_dict[confid],
                    alpha=0.5,
                    lw=0,
                    fill=True,
                    zorder=-2,
                )
            )
        plt.yticks(
            ticks=2 * confid_indices,
            labels=list(colors_dict.keys()),
            fontsize=23,
        )
        plt.xlim(-0.6, 2 * n_confid - 0.4)
        plt.ylim(-0.6, 2 * n_confid - 0.4)
        plt.savefig(out_dir / "significance_maps_color_legend.pdf", bbox_inches="tight")
        plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()

    if no_labels:
        # with open(out_dir / "significance_maps_color_legend.txt", "w") as file:
        #     for k, v in colors_dict.items():
        #         file.write(f"{k}\t{v}\n")

        if flip_horizontally:
            for i, (c, confid) in enumerate(zip(significance, confid_names)):
                for j, s in enumerate(c):
                    if i == j:
                        continue

                    if s:
                        ax.add_patch(
                            Rectangle(
                                # (i - 0.5, j - 0.5),
                                (j - 0.5, n_confid - 1 - i - 0.5),
                                width=1,
                                height=1,
                                color=colors_dict[confid],
                                alpha=0.5,
                                lw=0,
                                fill=True,
                                zorder=-2,
                            )
                        )
                    else:
                        ax.scatter(
                            # [i],
                            # [j],
                            [j],
                            [n_confid - 1 - i],
                            marker="X",
                            s=300,
                            c=colors_dict[confid],
                            alpha=0.5,
                        )
            # plt.plot([-0.6, n_confid-0.4], [-0.6, n_confid-0.4], color="k", lw=0.5, zorder=10)
            plt.plot(
                [-0.6, n_confid - 0.4],
                [n_confid - 0.4, -0.6],
                color="k",
                lw=0.5,
                zorder=10,
            )
            plt.grid(color="whitesmoke")

            plt.gca().tick_params(axis="x", which="minor", bottom=False)
            plt.xticks(ticks=confid_indices, labels=n_confid * [])
            plt.gca().tick_params(axis="y", which="minor", left=False)
            plt.yticks(ticks=confid_indices, labels=n_confid * [])
            plt.gca().tick_params(axis="y", which="major", left=False, right=True)

        else:
            for i, (c, confid) in enumerate(zip(significance, confid_names)):
                for j, s in enumerate(c):
                    if i == j:
                        continue

                    if s:
                        ax.add_patch(
                            Rectangle(
                                # (i - 0.5, j - 0.5),
                                (n_confid - 1 - j - 0.5, n_confid - 1 - i - 0.5),
                                width=1,
                                height=1,
                                color=colors_dict[confid],
                                alpha=0.5,
                                lw=0,
                                fill=True,
                                zorder=-2,
                            )
                        )
                    else:
                        ax.scatter(
                            # [i],
                            # [j],
                            [n_confid - 1 - j],
                            [n_confid - 1 - i],
                            marker="X",
                            s=300,
                            c=colors_dict[confid],
                            alpha=0.5,
                        )

            plt.plot(
                [-0.6, n_confid - 0.4],
                [-0.6, n_confid - 0.4],
                color="k",
                lw=0.5,
                zorder=10,
            )
            plt.grid(color="whitesmoke")

            plt.gca().tick_params(axis="x", which="minor", bottom=False)
            plt.xticks(ticks=confid_indices, labels=n_confid * [])
            plt.gca().tick_params(axis="y", which="minor", left=False)
            plt.yticks(ticks=confid_indices, labels=n_confid * [])

    else:
        for i, (c, confid) in enumerate(zip(significance, confid_names)):
            for j, s in enumerate(c):
                if i == j:
                    continue

                if s:
                    ax.add_patch(
                        Rectangle(
                            (i - 0.5, j - 0.5),
                            width=1,
                            height=1,
                            color=colors_dict[confid],
                            alpha=0.5,
                            lw=0,
                            fill=True,
                            zorder=-2,
                        )
                    )
                else:
                    ax.scatter(
                        [i],
                        [j],
                        marker="X",
                        s=300,
                        c=colors_dict[confid],
                        alpha=0.5,
                    )

        plt.plot(
            [-0.6, n_confid - 0.4], [-0.6, n_confid - 0.4], color="k", lw=0.5, zorder=10
        )
        plt.grid(color="whitesmoke")

        plt.xticks(
            ticks=confid_indices + 0.4,
            labels=list(histograms.index),
            rotation=60,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=23,
            minor=True,
        )
        plt.yticks(
            ticks=confid_indices + 0.25,
            labels=list(histograms.index),
            rotation=30,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=23,
            minor=True,
        )

        plt.gca().tick_params(axis="x", which="minor", bottom=False)
        plt.xticks(ticks=confid_indices, labels=n_confid * [])
        plt.gca().tick_params(axis="y", which="minor", left=False)
        plt.yticks(ticks=confid_indices, labels=n_confid * [])

    plt.xlim(-0.6, n_confid - 0.4)
    plt.ylim(-0.6, n_confid - 0.4)
    # plt.title(filename)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()


def bs_kendall_tau_violin(
    data: pd.DataFrame,
    metric: str,
    histograms: pd.DataFrame,
    out_dir: Path,
    filename: str,
) -> None:
    """"""
    n_confid = histograms.shape[0]
    rank_values = np.arange(1, n_confid + 1)

    # Reindex columns handling the case of shared last ranks
    histograms = histograms.reindex(columns=rank_values, fill_value=0)
    confid_to_rank_idx = {histograms.index[i]: i for i in range(n_confid)}

    taus = []
    taus_aurc_augrc = []

    data["rank_2"] = data.groupby(
        ["bootstrap_index", "run"] if "run" in data.columns else ["bootstrap_index"]
    )["aurc" if metric == "augrc" else "augrc"].rank(method="min")

    for (_, group), (_, group_2) in zip(
        data.sort_values(by="rank").groupby(
            "bootstrap_index"
            if "run" not in data.columns
            else ["bootstrap_index", "run"]
        ),
        data.sort_values(by="rank_2").groupby(
            "bootstrap_index"
            if "run" not in data.columns
            else ["bootstrap_index", "run"]
        ),
    ):
        if len(group) != n_confid:
            raise ValueError(
                f"Missing results for the following group (expected {n_confid} "
                f"confids):\n\n{group}"
            )

        taus.append(
            kendalltau(
                x=np.arange(n_confid),
                y=[confid_to_rank_idx[c] for c in group["confid"]],
            ).statistic
        )

        taus_aurc_augrc.append(
            kendalltau(
                x=[confid_to_rank_idx[c] for c in group["confid"]],
                y=[confid_to_rank_idx[c] for c in group_2["confid"]],
            ).statistic
        )

    plt.violinplot(
        dataset=taus,
        positions=[1],
        showextrema=False,
        showmedians=False,
    )
    plt.boxplot(
        x=taus,
        positions=[1],
    )

    plt.violinplot(
        dataset=taus_aurc_augrc,
        positions=[2],
        showextrema=False,
        showmedians=False,
    )
    plt.boxplot(
        x=taus_aurc_augrc,
        positions=[2],
    )

    plt.grid(color="gray", lw=1.5, alpha=0.15)
    plt.xticks(ticks=[1, 2], labels=[f"{metric.upper()} stability", "AUGRC vs. AURC"])
    plt.ylabel("Kendall's tau")
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()


def bs_kendall_tau_comparing_metrics(
    data: dict,
    histograms: dict,
    out_dir: Path,
    filename: str,
) -> None:
    """"""
    data_aurc = data["aurc"]
    data_augrc = data["augrc"]
    histograms_aurc = histograms["aurc"]
    histograms_augrc = histograms["augrc"]

    n_confid = histograms_augrc.shape[0]
    rank_values = np.arange(1, n_confid + 1)

    # Reindex columns handling the case of shared last ranks
    histograms_aurc = histograms_aurc.reindex(columns=rank_values, fill_value=0)
    histograms_augrc = histograms_augrc.reindex(columns=rank_values, fill_value=0)

    confid_to_rank_idx_aurc = {histograms_aurc.index[i]: i for i in range(n_confid)}
    confid_to_rank_idx_augrc = {histograms_augrc.index[i]: i for i in range(n_confid)}

    taus_aurc = []
    taus_augrc = []
    taus_aurc_augrc = []

    for (l1, group_aurc), (l2, group_augrc) in zip(
        data_aurc.sort_values(by="rank").groupby(
            ["bootstrap_index", "study"]
            if "run" not in data_aurc.columns
            else ["bootstrap_index", "run", "study"]
        ),
        data_augrc.sort_values(by="rank").groupby(
            ["bootstrap_index", "study"]
            if "run" not in data_augrc.columns
            else ["bootstrap_index", "run", "study"]
        ),
    ):
        taus_aurc.append(
            kendalltau(
                x=np.arange(n_confid),
                y=[confid_to_rank_idx_aurc[c] for c in group_aurc["confid"]],
            ).statistic
        )
        taus_augrc.append(
            kendalltau(
                x=np.arange(n_confid),
                y=[confid_to_rank_idx_augrc[c] for c in group_augrc["confid"]],
            ).statistic
        )
        taus_aurc_augrc.append(
            kendalltau(
                x=[confid_to_rank_idx_aurc[c] for c in group_aurc["confid"]],
                y=[confid_to_rank_idx_augrc[c] for c in group_augrc["confid"]],
            ).statistic
        )

    # AURC
    plt.violinplot(
        dataset=taus_aurc,
        positions=[1],
        showextrema=False,
        showmedians=False,
    )
    plt.boxplot(
        x=taus_aurc,
        positions=[1],
    )
    # AUGRC
    plt.violinplot(
        dataset=taus_augrc,
        positions=[2],
        showextrema=False,
        showmedians=False,
    )
    plt.boxplot(
        x=taus_augrc,
        positions=[2],
    )
    # Comparison
    plt.violinplot(
        dataset=taus_aurc_augrc,
        positions=[3],
        showextrema=False,
        showmedians=False,
    )
    plt.boxplot(
        x=taus_aurc_augrc,
        positions=[3],
    )

    plt.ylim(0.48, 1.01)

    plt.grid(color="gray", lw=1.5, alpha=0.15)
    plt.xticks(
        ticks=[1, 2, 3], labels=[f"AURC stability", "AUGRC stability", "AUGRC vs. AURC"]
    )
    plt.ylabel("Kendall's tau")
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()
