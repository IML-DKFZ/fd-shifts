from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich import print


@dataclass
class Experiment:
    dataset: str
    model: str
    bb: str
    lr: str
    bs: str
    run: str
    do: str
    rew: str

    @staticmethod
    def from_standardized_name(name: str) -> Experiment:
        dataset = name.split("_")[0]

        # to also catch wilds_animals and wilds_camelyon and super_cifar100
        if dataset in ["wilds", "super"]:
            dataset += "_" + name.split("_")[1]

        # catch openset runs
        if "openset" in name:
            dataset += "_openset"

        return Experiment(
            dataset=dataset,
            model=extract_hparam(name, r"model([a-z]+)", "vit"),
            bb=extract_hparam(name, r"bb([a-z0-9]+(_small_conv)?)", "vit"),
            lr=extract_hparam(name, r"lr([0-9.]+)", None),
            bs=extract_hparam(name, r"bs([0-9]+)", "128"),
            run=extract_hparam(name, r"run([0-9]+)", "0"),
            do=extract_hparam(name, r"do([01])", "0"),
            rew=extract_hparam(name, r"rew([0-9.]+)", "0"),
        )


def extract_hparam(name: str, regex: str, default: str | None = None) -> str:
    if hparam := re.search(regex, name):
        return hparam[1]

    if default is None:
        raise ValueError(
            f"Value with regex {regex} could not be found and no default provided"
        )

    return default


def to_experiment_list(paths):
    result = pd.DataFrame([Experiment.from_standardized_name(path) for path in paths])
    result = result.drop_duplicates(["dataset", "model", "bb", "lr", "bs", "do", "rew"])
    result = (
        result.groupby(["dataset", "model", "do"])
        .aggregate(lambda s: list({i for i in s}))
        .reset_index()
    )
    result[["dataset", "model", "do"]] = result[["dataset", "model", "do"]].applymap(
        lambda x: [x]
    )
    return result[["dataset", "model", "bb", "lr", "bs", "do", "rew"]]


def to_experiment_definition(row):
    #     dset  model bb    lr    bsize do    reward    runs            confid stages
    # tuple[list, list, list, list, list, list, list, Union[range, list], Optional[list]]
    return str(tuple(row) + (range(5), [None]))


def main():
    base_path = Path("~/Experiments/vit").expanduser()
    paths = base_path.glob("**/test_results/raw_output.npz")
    paths = sorted(
        set(
            filter(
                lambda x: "vit" in str(x.relative_to(base_path)),
                map(lambda p: p.parent.parent, paths),
            )
        )
    )

    df = to_experiment_list(map(lambda x: str(x.relative_to(base_path)), paths))

    for _, experiment in df.iterrows():
        print(to_experiment_definition(experiment) + ",")


if __name__ == "__main__":
    main()
