from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from rich import print

pd.set_option("display.precision", 12)


def maximum_softmax_probability(softmax: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.max(softmax, axis=1)


def predictive_entropy(softmax: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.sum(softmax * (-np.log(softmax + 1e-7)), axis=1)


def predictive_entropy_log(softmax: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.sum(np.exp(softmax) * (-softmax), axis=1)


base_path = Path("~/Experiments/vit/").expanduser()
runs = [
    # "wilds_camelyon_lr0.001_run1_do0",
    "wilds_camelyon_modelvit_bbvit_lr0.001_bs128_run1_do0_rew0",
    "wilds_camelyon_modelvit_bbvit_lr0.001_bs128_run6_do0_rew0",
    # "wilds_camelyon_lr0.003_run0_do1",
    # "cifar10_lr0.0003_run1",
    # "cifar10_lr0.01_run0_do1",
    # "cifar100_lr0.01_run0_do1",
    # "cifar100_lr0.03_run1_do0",
    # "breeds_lr0.001_run0",
    # "breeds_lr0.01_run1_do1",
    # "svhn_lr0.01_run1_do0",
    # "svhn_lr0.01_run0_do1",
    # "wilds_animals_lr0.001_run1",
    # "wilds_animals_lr0.01_run0_do1",
    # "svhn_openset_modelvit_bbvit_lr0.01_bs128_run0_do0_rew0",
    # "svhn_openset_modelvit_bbvit_lr0.01_bs128_run0_do1_rew0",
    # "wilds_animals_openset_modelvit_bbvit_lr0.001_bs128_run3_do0_rew0",
    # "wilds_animals_openset_modelvit_bbvit_lr0.01_bs128_run3_do1_rew0",
]

# base_path = Path("~/Experiments/fd-shifts/camelyon_paper_sweep/").expanduser()
# runs = [f"confidnet_bbresnet50_do0_run{i}_rew2.2" for i in range(1, 10)]

report = []

for run in runs:
    output = np.load(base_path / run / "test_results/raw_output.npz")["arr_0"]
    print(output.dtype)

    df = pd.DataFrame(
        output,
        columns=[
            *(("softmax", i) for i in range(output.shape[1] - 2)),
            ("label", ""),
            ("dataset", ""),
        ],
        dtype=np.float64,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    softmax = df.softmax.to_numpy(dtype=np.float64)[:10]
    # msr = maximum_softmax_probability(softmax)
    # pe = predictive_entropy(softmax)
    # print(np.argsort(msr))
    # print(np.argsort(-pe))

    test_data = df[:10][["softmax", "label"]]

    test_data = test_data.assign(
        msr=maximum_softmax_probability(test_data.softmax),
        pe=predictive_entropy(test_data.softmax),
        error=(
            (test_data.softmax.to_numpy(dtype=np.float64).max(axis=1) == 1)
            & (
                (test_data.softmax.to_numpy(dtype=np.float64) > 0)
                & (test_data.softmax.to_numpy(dtype=np.float64) < 1)
            ).any(axis=1)
        ),
    )
    idx_sorted_msr = np.argsort(test_data.msr)
    test_data["msr_rank"] = idx_sorted_msr.sort_values().index
    idx_sorted_pe = np.argsort(-test_data.pe)
    test_data["pe_rank"] = idx_sorted_pe.sort_values().index
    print(test_data)

    total = df.shape[0]
    error_df = df[
        (df.softmax.max(axis=1) == 1)
        & ((df.softmax > 0) & (df.softmax < 1)).any(axis=1)
    ]

    error = error_df.shape[0]
    print(
        f"[red bold]{run + ':':70s}[/red bold][green italic]Error {error}/{total} or {error/total*100:.2f}%[/green italic]"
    )

    report.append([run, error, total])

pd.DataFrame(report, columns=("run", "error", "total")).to_csv(
    f"./{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}-numerical_error_samples.csv"
)
