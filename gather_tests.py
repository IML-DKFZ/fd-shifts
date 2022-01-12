import re
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
from rich import print

datasets = [
    "cifar10_",
    "cifar100",
    "super_cifar100",
    "breeds",
    "svhn",
    "wilds_animals",
    "wilds_camelyon",
]

experiments = [
    "vit",
    "vit_devries",
    "vit_dg",
    "vit_confidnet",
]


def check_missing_tests():
    rewards = [2.2, 3, 4.5, 6, 10]
    exps: list[
        tuple[list, list, list, list, list, list, list, range, Optional[list]]
    ] = [
        (
            ["cifar10"],
            ["confidnet"],
            ["vit"],
            [0.01],
            [256 // 2],
            [1],
            [2.2],
            range(1),
            [2],
        ),
        (["breeds"], ["dg"], ["vit"], [0.01], [128], [1], rewards, range(1), [None]),
        (["svhn"], ["dg"], ["vit"], [0.01], [128], [1], rewards, range(1), [None]),
        (
            ["wilds_camelyon"],
            ["dg"],
            ["vit"],
            [0.003],
            [128],
            [1],
            rewards,
            range(1),
            [None],
        ),
        (
            ["cifar10"],
            ["devries"],
            ["vit"],
            [0.0003],
            [128],
            [0],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["cifar10"],
            ["devries"],
            ["vit"],
            [0.01],
            [128],
            [1],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["breeds"],
            ["devries"],
            ["vit"],
            [0.001],
            [128],
            [0],
            [2.2],
            range(1),
            [None],
        ),
        (["breeds"], ["devries"], ["vit"], [0.01], [128], [1], [2.2], range(1), [None]),
        (["svhn"], ["devries"], ["vit"], [0.01], [128], [0], [2.2], range(1), [None]),
        (["svhn"], ["devries"], ["vit"], [0.01], [128], [1], [2.2], range(1), [None]),
        (
            ["wilds_camelyon"],
            ["devries"],
            ["vit"],
            [0.001],
            [128],
            [0],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["wilds_camelyon"],
            ["devries"],
            ["vit"],
            [0.003],
            [128],
            [1],
            [2.2],
            range(1),
            [None],
        ),
        (["cifar100"], ["dg"], ["vit"], [0.01], [512], [1], rewards, range(1), [None]),
        (
            ["wilds_animals"],
            ["dg"],
            ["vit"],
            [0.01],
            [512],
            [1],
            rewards,
            range(1),
            [None],
        ),
        (
            ["cifar100"],
            ["devries"],
            ["vit"],
            [0.03],
            [512],
            [0],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["wilds_animals"],
            ["devries"],
            ["vit"],
            [0.001],
            [512],
            [0],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["cifar100"],
            ["devries"],
            ["vit"],
            [0.01],
            [512],
            [1],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["wilds_animals"],
            ["devries"],
            ["vit"],
            [0.01],
            [512],
            [1],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["super_cifar100"],
            ["dg"],
            ["vit"],
            [0.001],
            [128],
            [1],
            rewards,
            range(1),
            [None],
        ),
        (
            ["super_cifar100"],
            ["devries"],
            ["vit"],
            [0.003],
            [128],
            [0],
            [2.2],
            range(1),
            [None],
        ),
        (
            ["super_cifar100"],
            ["devries"],
            ["vit"],
            [0.001],
            [128],
            [1],
            [2.2],
            range(1),
            [None],
        ),
    ]

    base_path = Path("~/cluster/experiments/vit").expanduser()

    for experiment in exps:
        for dataset, model, bb, lr, bs, do, rew, run, stage in product(*experiment):
            exp_name = "{}_model{}_bb{}_lr{}_bs{}_run{}_do{}_rew{}".format(
                dataset, model, bb, lr, bs, run, do, rew,
            )

            if not (base_path / exp_name).exists():
                print(
                    f"[bold blue]{exp_name} not found, might not have been run at all (or uses old naming scheme)"
                )
                continue

            if not (base_path / exp_name / "test_results").exists():
                print(f"[bold red]{exp_name}/test_results not found")


def rename(row):
    name = "vit"

    lr = re.search(r"lr([0-9.]+)", row)[1]
    name = f"{name}_lr{lr}"

    do = re.search(r"do([0-9.]+)", row)
    if do is None:
        do = 0
    else:
        do = do[1]
    name = f"{name}_do{do}"

    rew = re.search(r"rew([0-9.]+)", row)
    if rew is None:
        rew = 0
    else:
        rew = rew[1]
    name = f"{name}_rew{rew}"

    name = f"{name}_bbvit"

    run = re.search(r"run([0-9])", row)[1]
    name = f"{name}_run{run}"

    return name


def main():
    pd.set_option("display.max_rows", 999)

    for dataset in datasets:
        print(f"[bold]Experiment: [/][bold red]{dataset.replace('_', '')}[/]")
        print("[bold]Looking for test results...")

        base_path = Path("~/cluster/experiments/vit").expanduser()
        df = [
            pd.read_csv(p)
            for p in base_path.glob(f"{dataset}*_run*/test_results/*.csv")
        ]

        df = pd.concat(df)
        df = df[~df["study"].str.contains("224")]

        df["old_name"] = df["name"]
        df["name"] = df.name.apply(rename)

        def name_to_metric(metric: str, value: str):
            match = re.search(metric + r"([0-9.]+)", value)
            if match is not None:
                return match[1]

            return ""

        df["lr"] = df.name.apply(lambda value: name_to_metric("lr", value))
        df["do"] = df.name.apply(lambda value: name_to_metric("do", value))
        df["run"] = df.name.apply(lambda value: name_to_metric("run", value))
        df["rew"] = df.name.apply(lambda value: name_to_metric("rew", value))
        df["model"] = df.old_name.apply(
            lambda value: match[1]
            if (match := re.search(r"model([^_]+)", value))
            else "vit"
        )

        def select_func(row, selection_df, selection_column):
            if selection_column == "rew" and "dg" not in row.model:
                return 1

            if "det" in row.confid:
                row_confid = "det_"
            elif "mcd" in row.confid:
                row_confid = "mcd_"
            else:
                row_confid = ""

            if "maha" in row.confid:
                row_confid = row_confid + "maha"
            elif "dg" in row.model:
                row_confid = "dg"
            elif "devries" in row.confid:
                row_confid = row_confid + "devries"
            elif "tcp" in row.confid:
                row_confid = row_confid + "tcp"
            else:
                row_confid = row_confid + "pe"

            if selection_column == "rew":
                row_confid = "dg"

            selection_df = selection_df[
                (selection_df.confid == row_confid) & (selection_df.do == row.do)
            ]

            try:
                if row[selection_column] == selection_df[selection_column].tolist()[0]:
                    return 1
                else:
                    return 0
            except IndexError as e:
                print(f"{dataset} {row_confid} {row}")
                raise e

        # Select best single run lr based on metric
        metric = "aurc"
        selection_df = df[(df.study == "val_tuning")][
            ["name", "confid", "lr", "do", "run", metric]
        ]
        selection_df = selection_df[
            (selection_df.confid.str.contains("pe"))
            | (selection_df.confid.str.contains("maha"))
            | (selection_df.confid.str.contains("dg"))
            | (selection_df.confid.str.contains("devries"))
            | (selection_df.confid.str.contains("tcp"))
            & (~(selection_df.confid.str.contains("waic")))
        ].reset_index()
        selection_df.confid = selection_df.confid.str.replace("maha_mcd", "mcd_maha")
        selection_df.confid = selection_df.confid.str.replace("dg_mcd", "mcd_dg")
        selection_df.confid = selection_df.confid.str.replace(
            "devries_mcd", "mcd_devries"
        )
        selection_df.confid = selection_df.confid.str.replace("tcp_mcd", "mcd_tcp")
        selection_df = selection_df.iloc[
            selection_df.groupby(["confid", "do"])[metric].idxmin()
        ]
        # print(selection_df)

        df["select_lr"] = df.apply(
            lambda row: select_func(row, selection_df, "lr"), axis=1
        )

        # Select best single run rew based on metric
        metric = "aurc"
        selection_df = df[(df.study == "val_tuning") & (df.select_lr == 1)][
            ["name", "confid", "rew", "do", "run", metric]
        ]
        # selection_df = selection_df[df[(df.study == "val_tuning")]["select_lr"] == 1]
        selection_df = selection_df[
            (selection_df.confid == "dg")
            & (~(selection_df.confid.str.contains("waic")))
        ].reset_index()
        selection_df.confid = selection_df.confid.str.replace("maha_mcd", "mcd_maha")
        selection_df.confid = selection_df.confid.str.replace("dg_mcd", "mcd_dg")
        selection_df.confid = selection_df.confid.str.replace(
            "devries_mcd", "mcd_devries"
        )
        selection_df.confid = selection_df.confid.str.replace("tcp_mcd", "mcd_tcp")
        selection_df = selection_df.iloc[
            selection_df.groupby(["confid", "do"])[metric].idxmin()
        ]
        # print(selection_df)

        df["select_rew"] = df.apply(
            lambda row: select_func(row, selection_df, "rew"), axis=1
        )

        # print(df)
        selected_df = df[(df.select_lr == 1) & (df.select_rew == 1)]
        # selected_df = df[(df.select_lr == 1)]
        potential_runs = (
            selected_df[(selected_df.study == "iid_study")][
                ["model", "lr", "run", "do", "rew",]
            ]
            .drop_duplicates()
            .groupby(["model", "lr", "do", "rew",])
            .max()
            .reset_index()
        )

        # print(potential_runs)
        for run in potential_runs.itertuples():
            # print(run)
            print(f'(["{dataset}"], ["{run.model}"], ["vit"], [{run.lr}], [128], [{run.do}], [{run.rew}], range({int(run.run) + 1}, 5), [1, 2]),')

        dataset = dataset.replace("wilds_", "")
        dataset = dataset.replace("cifar10_", "cifar10")
        # print(selected_df)

        out_path = Path("~/Projects/failure-detection-benchmark/results").expanduser()
        selected_df.to_csv(out_path / f"{dataset}vit.csv")


if __name__ == "__main__":
    main()
    # check_missing_tests()
