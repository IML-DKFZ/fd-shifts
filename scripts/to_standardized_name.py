from __future__ import annotations

import re
from pathlib import Path

from rich import print


def extract_hparam(name: str, regex: str, default: str | None = None) -> str:
    if hparam := re.search(regex, name):
        return hparam[1]

    if default is None:
        raise ValueError(
            f"Value with regex {regex} could not be found and no default provided"
        )

    return default


def to_full_name(name: str) -> str:
    dataset = name.split("_")[0]

    # to also catch wilds_animals and wilds_camelyon and super_cifar100
    if dataset in ["wilds", "super"]:
        dataset += "_" + name.split("_")[1]

    # catch openset runs
    if "openset" in name:
        dataset += "_openset"

    return (
        f"{dataset}_"
        f"model{extract_hparam(name, r'model([a-z]+)', 'vit')}_"
        f"bb{extract_hparam(name, r'bb([a-z0-9]+(_small_conv)?)', 'vit')}_"
        f"lr{extract_hparam(name, r'lr([0-9.]+)', None)}_"
        f"bs{extract_hparam(name, r'bs([0-9]+)', '128')}_"
        f"run{extract_hparam(name, r'run([0-9]+)', '0')}_"
        f"do{extract_hparam(name, r'do([01])', '0')}_"
        f"rew{extract_hparam(name, r'rew([0-9.]+)', '0')}"
    )


base_path = Path("~/Experiments/vit").expanduser()
paths = base_path.glob("**/test_results/raw_output.npz")
paths = sorted(set(map(str, map(lambda p: p.relative_to(base_path).parent.parent, paths))))

new_paths = map(to_full_name, paths)

for old_name, new_name in zip(paths, new_paths):
    if old_name == new_name:
        continue
    
    for s in old_name.split("_"):
        assert s in new_name

    print(f"[red bold]{base_path / old_name}")
    print(f"[green bold]{base_path / new_name}")

    try:
        (base_path / old_name).rename(base_path / new_name)
    except:
        if not (base_path / new_name / "test_results/raw_output.npz").is_file():
            print(f"[blue bold]{base_path / new_name}")

    print()
