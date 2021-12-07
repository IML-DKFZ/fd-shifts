from pathlib import Path
import pandas as pd
from rich import print
from rich.table import Table
import re


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

        df["lr"] = df.name.apply(lambda row: re.search(r"lr([0-9.]+)", row)[1])
        df["do"] = df.name.apply(lambda row: re.search(r"do([0-9]+)", row)[1])
        df["run"] = df.name.apply(lambda row: re.search(r"run([0-9]+)", row)[1])

        def select_func(row, selection_df, selection_column):
            if "det" in row.confid:
                row_confid = "det_"
            elif "mcd" in row.confid:
                row_confid = "mcd_"
            else:
                row_confid = ""

            if "maha" in row.confid:
                row_confid = row_confid + "maha"
            elif "dg" in row.confid:
                row_confid = row_confid + "dg"
            elif "devries" in row.confid:
                row_confid = row_confid + "devries"
            else:
                row_confid = row_confid + "pe"

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
            & (~(selection_df.confid.str.contains("waic")))
        ].reset_index()
        selection_df.confid = selection_df.confid.str.replace("maha_mcd", "mcd_maha")
        selection_df.confid = selection_df.confid.str.replace("dg_mcd", "mcd_dg")
        selection_df.confid = selection_df.confid.str.replace("devries_mcd", "mcd_devries")
        selection_df = selection_df.iloc[
            selection_df.groupby(["confid", "do"])[metric].idxmin()
        ]
        # print(selection_df)

        df["select_lr"] = df.apply(
            lambda row: select_func(row, selection_df, "lr"), axis=1
        )

        # print(df)
        selected_df = df[df.select_lr == 1]
        print(
            selected_df[
                (selected_df.confid == "det_pe") & (selected_df.study == "iid_study")
            ][
                [
                    "name",
                    "accuracy",
                    "failauc",
                    "aurc",
                    "ece",
                    "fail-NLL",
                    "lr",
                    "do",
                    "run",
                    "old_name",
                ]
            ].sort_values(["do", "run"])
        )

        dataset = dataset.replace("wilds_", "")
        dataset = dataset.replace("cifar10_", "cifar10")
        print(selected_df)

        # # De Vries
        # base_path = Path("~/cluster/experiments/").expanduser()
        # csv2 = [pd.read_csv(p) for p in base_path.glob(f"devries_vit_{model}/*_run*/test_results/*.csv")]
        # if len(csv2) > 0:
        #     print(f"Found {len(csv2)} De Vries runs")
        #     csv2 = pd.concat(csv2)
        #     csv2["name"] = csv2.apply(lambda row: row["name"] + "_rew0", axis=1)
        #
        #     print(model)
        #     csv = pd.concat([csv, csv2])

        out_path = Path("~/Projects/failure-detection-benchmark/results").expanduser()
        selected_df.to_csv(out_path / f"{dataset}vit.csv")


if __name__ == "__main__":
    main()
