import shutil
from pathlib import Path

import yaml
from rich import print  # pylint: disable=redefined-builtin


def main():
    base_path = Path("~/Experiments/vit_64").expanduser()
    base_path32 = Path("~/Experiments/vit_32/").expanduser()

    for path in base_path.glob("*openset*"):
        print(f"[bold blue]{path}[/bold blue]")
        versions = sorted((base_path32 / path.relative_to(base_path)).glob("*version*/last.ckpt"))
        print(len(versions))
        if len(versions) == 0:
            print(f"[bold red]{path}[/bold red]")
            continue

        latest: Path = base_path / versions[-1].parent.relative_to(base_path32)

        with open(latest / "hparams.yaml") as file:
            hparams = yaml.safe_load(file.read())

        out_classes = hparams["cf"]["data"]["kwargs"]["out_classes"]

        with open(path / "hydra/config.yaml") as file:
            hydra_params = yaml.safe_load(file.read())

        hydra_params["data"]["kwargs"]["out_classes"] = out_classes

        shutil.copy(path / "hydra/config.yaml", path / "hydra/config.yaml.bak")

        with open(path / "hydra/config.yaml", "w") as file:
            file.write(yaml.safe_dump(hydra_params))


if __name__ == "__main__":
    main()
