import re
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
from rich import print  # pylint: disable=redefined-builtin

pd.set_option("display.max_rows", None)

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

        df = pd.concat(df_list).reset_index(drop=True)
        df = df[~(df["confid"] == "det_pe")]
        df = df[~(df["confid"] == "det_mcp")]

        # if dataset == "svhn":
        #     df = df[~(df["confid"] == "dg")]
        #     df = df[~(df["confid"] == "dg_mcd")]
        #     df = df[~(df["confid"] == "devries")]
        #     df = df[~(df["confid"] == "tcp")]
        #     df = df[~(df["confid"] == "mcd_ee")]
        #     df = df[~(df["confid"] == "mcd_pe")]
        #     df = df[~(df["confid"] == "mcd_mcp")]

        ###
        base_path64 = Path("~/Experiments/fd-shifts_64").expanduser()
        paths64 = list(base_path64.glob(f"{dataset}*/*_run*/test_results/*.csv"))
        print(len(paths64))
        paths64.extend(list(base_path64.glob(f"multistep_{dataset}*/*_run*/test_results/*.csv")))
        print(len(paths64))

        if "openset" not in dataset:
            paths64 = filter(lambda x: "openset" not in str(x), paths64)

        try:
            df64 = pd.concat([pd.read_csv(p) for p in paths64])

            df = pd.concat([df, df64])
            if dataset == "svhn":
                df = df64
        except:
            pass
        ###

        if dataset == "svhn":
            # df = df[~(df.confid.str.contains("mcd_pe")) | ~(df.name.str.startswith("confidnet"))].reset_index(drop=True)
            print(df[df.study.str.contains("iid") & df.confid.str.contains("mcd_pe")][["name", "model", "aurc"]].sort_values("aurc"))
            # print(list(df[df.study.str.contains("iid") & df.confid.str.contains("det_pe") & df.model.str.contains("confidnet")].sort_values("aurc").index))
            # df = df.drop(list(df[df.study.str.contains("iid") & df.confid.str.contains("mcd_pe") & df.model.str.contains("confidnet")].sort_values("aurc").index)[-1:])
            print(df[df.study.str.contains("iid") & df.confid.str.contains("mcd_pe")][["name", "model", "aurc"]].sort_values("aurc"))

        if dataset == "cifar10_":
            print(df[df.study.str.contains("svhn") & df.confid.str.contains("det_pe")][["name", "model", "failauc"]].sort_values("failauc"))

        if dataset == "camelyon":
            print(
                df[
                    df.study.str.startswith("iid")
                    & (
                        df.confid.str.startswith("det_pe")
                        | df.confid.str.startswith("det_mcp")
                    )
                    & df.name.str.startswith("confidnet")
                ].sort_values(["name", "confid"])[["name", "confid", "failauc", "accuracy", "aurc"]]
            )


        dataset = dataset.replace("wilds_", "")
        dataset = dataset.replace("cifar10_", "cifar10")
        dataset = dataset.replace("supercifar", "super_cifar100")

        out_path = Path("~/Projects/failure-detection-benchmark/results64").expanduser()
        out_path.mkdir(exist_ok=True)
        df.to_csv(out_path / f"{dataset}.csv")


if __name__ == "__main__":
    main()
