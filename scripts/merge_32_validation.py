from pathlib import Path
from rich import print
import numpy as np


base_path = Path("~/Experiments").expanduser()
list32 = list((base_path / "vit_32").glob("*/test_results/*.npz"))
validation = list((base_path / "vit_32").glob("*/validation/*.npz"))
list64 = list((base_path / "vit_64").glob("*/test_results/*.npz"))


def to_base_name(p):
    return p.relative_to(base_path / "vit_32").parent.parent


def to_base_name64(p):
    return p.relative_to(base_path / "vit_64").parent.parent


# print(len(list32))
# list32 = list(filter(lambda p: to_base_name(p) in map(to_base_name, validation), list32))
# print(len(list32))
# print(len(validation))



sanity = list(map(lambda p: p.relative_to(base_path / "vit_64").parent.parent, list64))
list32 = list(filter(lambda p: to_base_name(p) in map(to_base_name64, list64), list32))
print(len(list32))

for p in list32:
    with np.load(p) as f:
        n32 = f["arr_0"].shape[0]

    # val = np.load(base_path / "vit_32" / to_base_name(p) / "validation/raw_output.npz")["arr_0"].shape[0]

    with np.load(base_path / "vit_64" / to_base_name(p) / "test_results/raw_output.npz") as f:
        n64 = f["arr_0"].shape[0]

    print(n32 == n64)
