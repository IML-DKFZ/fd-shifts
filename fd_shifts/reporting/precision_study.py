from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as spc
import torch
from rich.pretty import pprint

from fd_shifts import experiments

BASE_PATH = Path("/home/t974t/Experiments")


def get_experiments() -> list[experiments.Experiment]:
    _experiments = experiments.get_all_experiments()

    return list(
        filter(
            lambda experiment: "precision_study" in str(experiment.group_dir),
            _experiments,
        )
    )


def get_error_rate(experiment: experiments.Experiment):
    data = np.load(
        BASE_PATH / experiment.to_path() / "test_results" / "raw_logits.npz"
    )["arr_0"][:, :-2]
    softmax = spc.softmax(data, axis=1)
    msr = softmax.max(axis=1)
    errors = (msr == 1) & ((softmax > 0) & (softmax < 1)).any(axis=1)
    return errors.mean() * 100


def load_metrics() -> pd.DataFrame:
    datas = []
    for experiment in get_experiments():
        paths = list((BASE_PATH / experiment.to_path()).glob("**/*.csv"))
        if len(paths) == 0:
            continue
        data = pd.concat(
            map(pd.read_csv, paths)
        )
        data = data.assign(
            dataset=experiment.dataset,
            precision=str(experiment.group_dir.stem)[-2:] + "bit",
            dropout=experiment.dropout,
            run=experiment.run,
            error_rate=get_error_rate(experiment),
        )
        data = data[data.study == "iid_study"]
        data = data[data.dropout == 1]
        data = data[
            [
                "dataset",
                "study",
                "confid",
                "aurc",
                "failauc",
                "accuracy",
                "error_rate",
                "precision",
                "run",
                "dropout",
            ]
        ]
        datas.append(data)

    return pd.concat(datas)


def precision_study():
    data = load_metrics()
    fixed_columns = [
        "dataset",
        "study",
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
    # pprint(data)
    mean = data.groupby(fixed_columns).mean().reset_index()
    std = data.groupby(fixed_columns).std().reset_index()
    pprint(data[data.confid == "det_mcp"])

    fig, axs = plt.subplots(2, 4, figsize=(4*8, 2*6))
    fig: plt.Figure = fig
    axs: list[list[plt.Axes]] = axs

    for axes, dset in zip(axs, ["svhn", "camelyon"]):
        _mean = mean[mean.dataset == dset]
        _std = std[std.dataset == dset]
        axes[0].errorbar(
            _mean[_mean.confid == "det_mcp"].precision,
            _mean[_mean.confid == "det_mcp"].aurc,
            yerr=_std[_std.confid == "det_mcp"].aurc,
            label="Maximum Softmax Response",
            capsize=5,
        )
        axes[0].errorbar(
            _mean[_mean.confid == "det_pe"].precision,
            _mean[_mean.confid == "det_pe"].aurc,
            yerr=_std[_std.confid == "det_pe"].aurc,
            label="Predictive Entropy",
            capsize=5,
        )
        axes[0].set_ylabel("AURC")
        axes[0].set_xlabel("Precision")
        axes[0].set_title(dset)
        axes[0].legend()

        axes[1].errorbar(
            _mean[_mean.confid == "det_mcp"].precision,
            _mean[_mean.confid == "det_mcp"].failauc,
            yerr=_std[_std.confid == "det_mcp"].failauc,
            label="Maximum Softmax Response",
            capsize=5,
        )
        axes[1].errorbar(
            _mean[_mean.confid == "det_pe"].precision,
            _mean[_mean.confid == "det_pe"].failauc,
            yerr=_std[_std.confid == "det_pe"].failauc,
            label="Predictive Entropy",
            capsize=5,
        )
        axes[1].set_ylabel("FailAUC")
        axes[1].set_xlabel("Precision")
        axes[1].legend()

        axes[2].errorbar(
            _mean[_mean.confid == "det_mcp"].precision,
            _mean[_mean.confid == "det_mcp"].error_rate,
            yerr=_std[_std.confid == "det_mcp"].error_rate,
            capsize=5,
        )
        axes[2].set_ylabel("Error Rate * 100")
        axes[2].set_xlabel("Precision")

        axes[3].errorbar(
            _mean[_mean.confid == "det_mcp"].precision,
            _mean[_mean.confid == "det_mcp"].accuracy,
            yerr=_std[_std.confid == "det_mcp"].accuracy,
            label="Maximum Softmax Response",
            capsize=5,
        )
        axes[3].errorbar(
            _mean[_mean.confid == "det_pe"].precision,
            _mean[_mean.confid == "det_pe"].accuracy,
            yerr=_std[_std.confid == "det_pe"].accuracy,
            label="Predictive Entropy",
            capsize=5,
        )
        axes[3].set_ylabel("Accuracy")
        axes[3].set_xlabel("Precision")
        axes[3].legend()

    fig.savefig("./test.png")


if __name__ == "__main__":
    precision_study()
