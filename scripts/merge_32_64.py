from pathlib import Path

from rich import print


def main():
    base_path = Path("~/Experiments").expanduser()
    list32 = list((base_path / "vit_32").glob("*/test_results/*.npz")) + list(
        (base_path / "vit_32").glob("*/hydra/*.yaml")
    )
    list64 = list((base_path / "vit_64").glob("*/test_results/*.npz"))

    def filter_unique(path: Path):
        return path.relative_to(base_path / "vit_32") not in list(
            map(lambda p: p.relative_to(base_path / "vit_64"), list64)
        )

    print("[bold red]List 32")
    print(len(list32))
    print(len(list(filter(filter_unique, list32))))

    print("[bold red]List 64")
    print(list64)


if __name__ == "__main__":
    main()
