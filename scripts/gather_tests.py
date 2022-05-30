import re
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
from rich import print  # pylint: disable=redefined-builtin

datasets = [
    "cifar10_",
    "cifar100",
    "super_cifar100",
    "breeds",
    "svhn",
    "wilds_animals",
    "wilds_camelyon",
    "svhn_openset",
    "wilds_animals_openset",
]


def rename(row_match: re.Match) -> str:
    # pylint: disable=invalid-name
    row: str = row_match[0]
    if not row:
        raise ValueError

    model = re.search(r"model([a-z]+)", row)
    if model is None:
        model = "vit"
    else:
        model = model[1]
    name = model

    lr = re.search(r"lr([0-9.]+)", row)
    assert lr
    lr = lr[1]
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

    if model == "confidnet":
        rew = 2.2
    name = f"{name}_rew{rew}"

    bb = re.search(r"bb([a-z]+)", row)
    if bb is None:
        bb = "vit"
    else:
        bb = bb[1]
    name = f"{name}_bb{bb}"

    run = re.search(r"run([0-9])", row)
    # assert run, f"{row}"
    if run is None:
        run = "0"
    else:
        run = run[1]
    name = f"{name}_run{run}"

    return name


def select_func(row, selection_df, selection_column):
    if selection_column == "rew" and "dg" not in row.model:
        return 1

    if "vit" not in row.bb:
        return 1

    if "det" in row.confid and not "maha" in row.confid:
        row_confid = "det_"
    elif "logits" in row.confid:
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

    selection_df_ = selection_df[
        (selection_df.confid == row_confid) & (selection_df.do == row.do)
    ]

    try:
        if row[selection_column] == selection_df_[selection_column].tolist()[0]:
            return 1
        return 0
    except IndexError as error:
        print(f"{row_confid} {row}")
        print(selection_df.confid)
        raise error


def main():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    for dataset in datasets:
        print(f"[bold]Experiment: [/][bold red]{dataset.replace('_', '')}[/]")
        print("[bold]Looking for test results...")

        base_path = Path("~/Experiments/vit_32").expanduser()

        paths = list(base_path.glob(f"{dataset}*run*/test_results/*.csv"))

        # TODO: Cannot filter based on csv age because of the new analysis
        paths = list(filter(
            lambda x: (
                # (("modeldg" not in str(x)) and ("modeldevries" not in str(x)))
                ("modeldg" not in str(x))
                or (
                    (x.parent / "raw_output.npz").stat().st_mtime
                    > datetime(2022, 1, 10).timestamp()
                )
            ),
            paths,
        ))
        tmp = paths.copy()
        paths = filter(lambda x: (x.parent / "analysis_metrics_val_tuning.csv") in tmp, paths)

        if "openset" not in dataset:
            paths = filter(lambda x: "openset" not in str(x), paths)

        paths = map(lambda x: (x, (x.parent / "raw_output.npz").stat().st_mtime), paths)

        df = [pd.read_csv(p).assign(date=datetime.fromtimestamp(d)) for p, d in paths]

        if len(df) < 1:
            continue

        print("[bold]Processing test results...")

        df = pd.concat(df)
        df = df[~df["study"].str.contains("224")]
        df = df[~(df["confid"] == "det_pe")]
        df = df[~(df["confid"] == "det_mcp")]

        ###
        base_path64 = Path("~/Experiments/vit_64").expanduser()

        paths64 = base_path64.glob(f"{dataset}*run*/test_results/*.csv")

        # TODO: Cannot filter based on csv age because of the new analysis
        # paths64 = filter(
        #     lambda x: (
        #         # (("modeldg" not in str(x)) and ("modeldevries" not in str(x)))
        #         ("modeldg" not in str(x))
        #         or (
        #             (x.parent / "raw_logits.npz").stat().st_mtime
        #             > datetime(2022, 1, 10).timestamp()
        #         )
        #     ),
        #     paths64,
        # )

        if "openset" not in dataset:
            paths64 = filter(lambda x: "openset" not in str(x), paths64)

        def to_equivalent_32_output(path: Path):
            # return (base_path / path.parent.parent.parts[-1] / "test_results" / "raw_output.npz")   
            return (base_path64 / path.parent.parent.parts[-1] / "test_results" / "raw_logits.npz")   

        paths64 = map(lambda x: (x, to_equivalent_32_output(x).stat().st_mtime), filter(lambda x: to_equivalent_32_output(x).is_file(), paths64))

        df64 = pd.concat([pd.read_csv(p).assign(date=datetime.fromtimestamp(d)) for p, d in paths64])
        print(df.confid.unique())

        df = pd.concat([df, df64])
        ###

        df["old_name"] = df["name"]
        df["name"] = df["name"].str.replace(".+", rename, regex=True)

        def name_to_metric(metric: str, value: str):
            return value.str.replace(r".*" + metric + r"([0-9.]+).*", "\\1", regex=True)

        df = df.assign(
            lr=lambda value: name_to_metric("lr", value.name),
            do=lambda value: name_to_metric("do", value.name),
            run=lambda value: name_to_metric("run", value.name),
            rew=lambda value: name_to_metric("rew", value.name),
            model=lambda value: value.old_name.str.replace(
                "(?:.*model([a-z]+))?.*", "\\1", regex=True
            ),
            bb=lambda value: value.old_name.str.replace(
                "(?:.*bb([a-z]+))?.*", "\\1", regex=True
            ),
        )

        df.model = df.model.replace("", "vit")
        df.bb = df.bb.replace("", "vit")

        # Select best single run lr based on metric
        metric = "aurc"
        selection_df = df[(df.study == "val_tuning")][
            ["name", "confid", "lr", "do", "run", "bb", metric]
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
            ["name", "confid", "rew", "do", "run", "bb", metric]
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
        selected_df = df[(df.select_lr == 1) & (df.select_rew == 1)].reset_index()
        if not "openset" in dataset:
            selected_df = selected_df[
                ~(
                    (selected_df.confid.str.contains("mcd"))
                    & ~((selected_df.model == "vit") | (selected_df.model == "dg"))
                )
            ]

        selected_df = selected_df[
            ~(
                (
                    (selected_df.confid.str.contains("det_mcp"))
                    | (selected_df.confid.str.contains("det_pe"))
                )
                & ~(selected_df.model == "vit")
            )
            | (selected_df.bb != "vit")
        ].reset_index(drop=True)

        if dataset == "svhn_openset":
            print(
                selected_df[
                    selected_df.study.str.startswith("iid")
                    & selected_df.confid.str.startswith("mcd_pe")
                    & selected_df.name.str.contains("bbvit")
                ].sort_values("aurc")[["name", "aurc", "date"]]
            )
            selected_df = selected_df[
                ~(
                    selected_df.name.str.startswith("vit")
                    & selected_df.confid.str.startswith("mcd_pe")
                    & selected_df.name.str.contains("bbvit")
                    & (selected_df.date > datetime(2022, 1, 22))
                )
            ]

        print(
            selected_df[
                selected_df.study.str.startswith("iid")
                & (
                    selected_df.confid.str.startswith("det_pe")
                    | selected_df.confid.str.startswith("det_mcp")
                )
            ].sort_values(["name", "confid"])[["confid", "aurc", "date", "old_name"]]
        )

        dataset = dataset.replace("wilds_", "")
        dataset = dataset.replace("cifar10_", "cifar10")
        print(len(selected_df))

        out_path = Path("~/Projects/failure-detection-benchmark/results64").expanduser()
        out_path.mkdir(exist_ok=True)
        selected_df[selected_df.bb == "vit"].to_csv(out_path / f"{dataset}vit.csv")
        if "openset" in dataset:
            selected_df[selected_df.bb != "vit"].to_csv(out_path / f"{dataset}.csv")


if __name__ == "__main__":
    main()
