import shutil
import subprocess
import tempfile
import typing
from itertools import chain, product
from pathlib import Path
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.special as spc
import torch
from matplotlib.figure import Figure
from rich.pretty import pprint

from fd_shifts import experiments, reporting
from fd_shifts.analysis import PlattScaling, confid_scores, metrics

BASE_PATH = Path("/media/experiments/")


def _get_experiments() -> list[experiments.Experiment]:
    _experiments = experiments.get_all_experiments()

    _experiments = list(
        filter(
            lambda experiment: "precision_study" not in str(experiment.group_dir),
            _experiments,
        )
    )
    _experiments = list(
        filter(lambda e: not (e.model != "vit" and e.backbone == "vit"), _experiments)
    )

    _experiments = list(filter(lambda e: e.dropout == 1, _experiments))

    res = list(
        chain(
            filter(
                lambda experiment: experiment.dataset == "svhn"
                and experiment.model == "confidnet",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "camelyon"
                and experiment.model == "confidnet",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "cifar10"
                and experiment.model == "confidnet",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "cifar100"
                and experiment.model == "confidnet",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "animals"
                and experiment.model == "confidnet",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "breeds"
                and experiment.model == "confidnet",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "svhn"
                and experiment.model == "vit"
                and experiment.learning_rate == 0.01,
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "wilds_camelyon"
                and experiment.model == "vit",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "cifar10"
                and experiment.model == "vit",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "svhn"
                and experiment.model == "vit",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "wilds_animals"
                and experiment.model == "vit",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "cifar100"
                and experiment.model == "vit",
                _experiments,
            ),
            filter(
                lambda experiment: experiment.dataset == "breeds"
                and experiment.model == "vit",
                _experiments,
            ),
        )
    )
    return res


def _get_error_rate(softmax: npt.NDArray):
    msr = softmax.max(axis=1)
    errors = msr == 1
    return errors.mean() * 100


def _print_32bit_error_16bit_ok(data: npt.NDArray):
    data32 = data.astype(np.float32)
    softmax32 = spc.softmax(data32, axis=1)
    msr32 = softmax32.max(axis=1)
    errors32 = (msr32 == 1) & ((softmax32 > 0) & (softmax32 < 1)).any(axis=1)

    data16 = data.astype(np.float16)
    softmax16 = spc.softmax(data16, axis=1)
    msr16 = softmax16.max(axis=1)
    errors16 = (msr16 == 1) & ((softmax16 > 0) & (softmax16 < 1)).any(axis=1)

    print(f"32bit Logits:\n{data32[errors32 & (~errors16)][0]}")
    print(f"16bit Logits:\n{data16[errors32 & (~errors16)][0]}")
    print(f"32bit Softmax:\n{softmax32[errors32 & (~errors16)][0]}")
    print(f"16bit Softmax:\n{softmax16[errors32 & (~errors16)][0]}")


def _load_metrics() -> pd.DataFrame:
    confids = ["det_mcp"]
    result_frame = []
    _dtype = {16: np.float16, 32: np.float32, 64: np.float64}
    for experiment in _get_experiments():
        path = BASE_PATH / experiment.to_path() / "test_results" / "raw_logits.npz"
        print(f"Loading {path}")

        try:
            data: npt.NDArray = np.load(path)["arr_0"]
        except (BadZipFile, FileNotFoundError):
            print(f"Failed loading {path}")
            continue

        dataset_idx = data[:, -1]
        label = data[:, -2]
        data = data[:, :-2]

        for precision in _dtype.keys():
            _data = data.astype(_dtype[precision])
            _softmax = spc.softmax(_data, axis=1)
            error_rate = _get_error_rate(_softmax)

            # only use iid test set (dataset_idx == 1)
            softmax = _softmax[dataset_idx == 1]
            _correct = (np.argmax(_softmax, axis=1) == label).astype(int)
            correct = _correct[dataset_idx == 1]

            for confid_type in confids:
                confid = confid_scores.get_confid_function(confid_type)(softmax)
                if "_pe" in confid_type:
                    confid = PlattScaling(
                        confid_scores.get_confid_function(confid_type)(
                            _softmax[dataset_idx == 0]
                        ),
                        _correct[dataset_idx == 0],
                    )(confid)
                stats_cache = metrics.StatsCache(confid, correct, 20)
                result_frame.append(
                    {
                        "model": "ViT" if experiment.model == "vit" else "CNN",
                        "dataset": experiment.dataset.replace("wilds_", "")
                        .upper()
                        .replace("CIFAR", "CIFAR-")
                        .replace("ANIMALS", "iWildCam"),
                        "confid": confid_type,
                        "aurc": metrics.aurc(stats_cache),
                        "failauc": metrics.failauc(stats_cache),
                        "accuracy": correct.mean(),
                        "error_rate": error_rate,
                        "precision": f"{precision}bit",
                        "run": experiment.run,
                        "dropout": experiment.dropout,
                    }
                )

    return pd.DataFrame.from_records(result_frame)


def precision_study(base_path: str | Path):
    """Create report table from precision study results

    Args:
        base_path (str | Path):
    """
    base_path = "./results"
    data_dir: Path = Path(base_path).expanduser().resolve()
    data = _load_metrics()
    fixed_columns = [
        "model",
        "dataset",
        "confid",
        "precision",
        "dropout",
    ]
    accum_columns = [
        "aurc",
        "failauc",
        "error_rate",
        "accuracy",
    ]
    mean = data.groupby(fixed_columns).mean().reset_index()
    std = data.groupby(fixed_columns).std().reset_index()

    out_table = pd.concat(
        [
            mean[(mean.confid == "det_mcp") & (mean.dropout == 1)].pivot(
                index=["model", "dataset"], columns="precision", values="error_rate"
            ),
            mean[(mean.confid == "det_mcp") & (mean.dropout == 1)].pivot(
                index=["model", "dataset"], columns="precision", values="aurc"
            ),
            mean[(mean.confid == "det_mcp") & (mean.dropout == 1)].pivot(
                index=["model", "dataset"], columns="precision", values="failauc"
            )
            * 100,
            mean[(mean.confid == "det_mcp") & (mean.dropout == 1)].pivot(
                index=["model", "dataset"], columns="precision", values="accuracy"
            )
            * 100,
        ],
        axis=1,
    )
    out_table.columns = pd.MultiIndex.from_product(
        [
            [
                "Round-to-one error rate $* 100 \\downarrow$",
                "AURC $* 10^3 \\downarrow$",
                "$\\mathrm{AUROC}_f * 100 \\uparrow$",
                "Accuracy",
            ],
            ["16bit", "32bit", "64bit"],
        ]
    )
    out_table = out_table.applymap(
        lambda x: f"{x:>3.3f}"[:5] if "." in f"{x:>3.3f}"[:4] else f"{x:>3.3f}"[:4],
    )
    out_table = out_table.drop(columns=[("Accuracy", "32bit"), ("Accuracy", "64bit")])
    out_table.columns = out_table.columns.map(
        lambda c: c
        if c != ("Accuracy", "16bit")
        else ("\\multicolumn{1}{c}{Accuracy $* 100 \\uparrow$}", "")
    )
    out_table.index.set_names(("", ""), inplace=True)

    dset_order = [
        "iWildCam",
        "BREEDS",
        "CAMELYON",
        "CIFAR-100",
        "CIFAR-10",
        "SVHN",
    ]

    print(out_table.index)

    def _key(t: pd.Index):
        return t.map(lambda x: dset_order.index(x))

    out_table = out_table.sort_index(level=1, key=_key)
    out_table = out_table.sort_index(level=0, sort_remaining=False)
    print(out_table.index)

    print(out_table)
    ltex = out_table.style.to_latex(
        convert_css=True,
        hrules=True,
        multicol_align=">{\centering}p{5.5cm}",
        column_format="ll?rrr??rrr?rrr?rrr?rrr?r",
    )

    # Remove toprule
    ltex = list(filter(lambda line: line != r"\toprule", ltex.splitlines()))

    # No separators in first header row
    ltex[1] = ltex[1].replace("?", "")
    del ltex[3]

    # Remove last separator in second header row
    # (this is just `replace("?", "", 1)`, but from the right)

    # Insert empty row before ViT part
    i = ltex.index(next((x for x in ltex if "ViT" in x)))
    ltex.insert(i, "\\midrule \\\\")

    ltex = "\n".join(ltex)

    with open(data_dir / f"precision_study.tex", "w") as f:
        f.write(ltex)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy2(
            data_dir / f"precision_study.tex",
            tmpdir / f"precision_study.tex",
        )
        with open(tmpdir / "render.tex", "w") as f:
            f.write(
                reporting.tables.LATEX_TABLE_TEMPLATE.replace(
                    "{input_file}", f"precision_study.tex"
                ).replace("{metric}", "precision study")
            )

        subprocess.run(f"lualatex render.tex", shell=True, check=True, cwd=tmpdir)
        shutil.copy2(
            tmpdir / "render.pdf",
            data_dir / f"precision_study.pdf",
        )

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(3 * 8, 2 * 6), squeeze=True)
    fig: Figure = fig

    axs: list[list[plt.Axes]] = list(map(list, zip(*axs)))

    for col, (axes, dset) in enumerate(zip(axs, ["camelyon", "cifar10", "svhn"])):
        _plot_data = out_table[
            (out_table.index.get_level_values(1) == dset)
            & (out_table.index.get_level_values(0) == "cnn")
        ]["Error Rate"]
        axes[0].plot(
            _plot_data.columns, _plot_data.to_numpy().flatten().astype(float), ".k--"
        )

        _plot_data = out_table[
            (out_table.index.get_level_values(1) == dset)
            & (out_table.index.get_level_values(0) == "vit")
        ]["Error Rate"]
        axes[0].plot(
            _plot_data.columns, _plot_data.to_numpy().flatten().astype(float), "^k:"
        )

        _plot_data = out_table[
            (out_table.index.get_level_values(1) == dset)
            & (out_table.index.get_level_values(0) == "cnn")
        ]["MSR AURC"]
        axes[1].plot(
            _plot_data.columns, _plot_data.to_numpy().flatten().astype(float), ".r--"
        )

        _plot_data = out_table[
            (out_table.index.get_level_values(1) == dset)
            & (out_table.index.get_level_values(0) == "cnn")
        ]["PE AURC"]
        axes[1].plot(
            _plot_data.columns, _plot_data.to_numpy().flatten().astype(float), ".b--"
        )

        _plot_data = out_table[
            (out_table.index.get_level_values(1) == dset)
            & (out_table.index.get_level_values(0) == "vit")
        ]["MSR AURC"]
        axes[1].plot(
            _plot_data.columns, _plot_data.to_numpy().flatten().astype(float), "^r:"
        )

        _plot_data = out_table[
            (out_table.index.get_level_values(1) == dset)
            & (out_table.index.get_level_values(0) == "vit")
        ]["PE AURC"]
        axes[1].plot(
            _plot_data.columns, _plot_data.to_numpy().flatten().astype(float), "^b:"
        )

        axes[0].set_title(dset)

        if col == 0:
            axes[0].set_ylabel("Error Rate [%]")
            axes[1].set_ylabel("AURC")

    (data_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(data_dir / "figures" / "precision_study.png")


if __name__ == "__main__":
    precision_study("./results")
