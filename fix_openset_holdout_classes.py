import shutil
from pathlib import Path

import yaml
from rich import print  # pylint: disable=redefined-builtin


def main():
    base_path = Path("~/results/vit").expanduser()

    for path in base_path.glob("*openset*"):
        versions = sorted(path.glob("*version*"))
        if len(versions) == 0:
            print(f"[bold red]{path}")
            continue

        latest: Path = versions[-1]

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
