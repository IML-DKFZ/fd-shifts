import re
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
from rich import print  # pylint: disable=redefined-builtin

datasets = [
    "cifar10_",
    "cifar100",
    "supercifar",
    "breeds",
    "svhn",
    "animals",
    "camelyon",
]


def main():

    for dataset in datasets:
        print(f"[bold]Experiment: [/][bold red]{dataset.replace('_', '')}[/]")
        print("[bold]Looking for test results...")

        base_path = Path("~/Experiments/fd-shifts/").expanduser()

        paths = list(base_path.glob(f"{dataset}*/*_run*/test_results/*.csv"))
        print(len(paths))
        paths.extend(list(base_path.glob(f"multistep_{dataset}*/*_run*/test_results/*.csv")))
        print(len(paths))

        if "openset" not in dataset:
            paths = filter(lambda x: "openset" not in str(x), paths)

        df_list = [pd.read_csv(p) for p in paths]

        print("[bold]Processing test results...")

        df = pd.concat(df_list)

        dataset = dataset.replace("wilds_", "")
        dataset = dataset.replace("cifar10_", "cifar10")
        dataset = dataset.replace("supercifar", "super_cifar100")

        out_path = Path("~/Projects/failure-detection-benchmark/results").expanduser()
        out_path.mkdir(exist_ok=True)
        df.to_csv(out_path / f"{dataset}.csv")


if __name__ == "__main__":
    main()
