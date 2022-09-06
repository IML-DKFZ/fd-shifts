from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from rich import print as pprint


@dataclass
class Experiment:
    group_dir: Path
    dataset: str
    model: str
    backbone: str
    dropout: int
    run: int
    reward: float
    learning_rate: float | None

    def to_string(self):
        pass

    def to_path(self):
        self.group_dir = Path(self.group_dir)
        if "vit" in str(self.group_dir):
            return self.group_dir / (
                f"{self.dataset}_"
                f"model{self.model}_"
                f"bb{self.backbone}_"
                f"lr{self.learning_rate}_"
                f"bs128_"
                f"run{self.run}_"
                f"do{self.dropout}_"
                f"rew{self.reward}"
            )

        if "precision_study" in str(self.group_dir.stem):
            return self.group_dir / (
                f"{self.model}_"
                f"bb{self.backbone}_"
                f"do{self.dropout}_"
                f"run{self.run + 1}_"
                f"rew{self.reward}"
            )

        return self.group_dir / (
            f"{self.dataset}_paper_sweep/"
            f"{self.model}_"
            f"bb{self.backbone}_"
            f"do{self.dropout}_"
            f"run{self.run + 1}_"
            f"rew{self.reward}"
        )

    @staticmethod
    def from_iterables(
        group_dir: Path,
        datasets: Iterable[str],
        models: Iterable[str],
        backbones: Iterable[str],
        dropouts: Iterable[int],
        runs: Iterable[int],
        rewards: Iterable[float],
        learning_rates: Iterable[float | None],
    ):
        return list(
            map(
                lambda args: Experiment(*args),
                product(
                    (group_dir,),
                    datasets,
                    models,
                    backbones,
                    dropouts,
                    runs,
                    rewards,
                    learning_rates,
                ),
            )
        )


def get_all_experiments(with_hyperparameter_sweep=True) -> list[Experiment]:
    _experiments = []

    # ViT Hyperparameter sweep
    if with_hyperparameter_sweep:
        _experiments.extend(
            Experiment.from_iterables(
                group_dir=Path("fd-shifts/vit"),
                datasets=(
                    "cifar10",
                    "cifar100",
                    "super_cifar100",
                    "svhn",
                    "breeds",
                    "wilds_animals",
                    "wilds_camelyon",
                ),
                models=("vit",),
                backbones=("vit",),
                dropouts=(0, 1),
                runs=(0,),
                rewards=(0,),
                learning_rates=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
            )
        )

    # ViT Best lr runs
    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("svhn",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(3e-2,),
            dropouts=(1,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("svhn",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("cifar10",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(1,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("cifar10",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(3e-4,),
            dropouts=(0,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("cifar100",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(1, 0),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("super_cifar100",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(3e-3,),
            dropouts=(0,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("super_cifar100",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-3,),
            dropouts=(1,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_animals",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-3,),
            dropouts=(0,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_animals",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(3e-3,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_camelyon",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-3,),
            dropouts=(0,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_camelyon",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(3e-3,),
            dropouts=(1,),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("breeds",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(3e-3, 1e-3),
            dropouts=(0,),
            runs=range(2),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("breeds",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(1,),
            runs=range(2),
            rewards=(0,),
        )
    )

    # Non-vit
    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("svhn",),
            models=("devries", "confidnet"),
            backbones=("svhn_small_conv",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("svhn",),
            models=("dg",),
            backbones=("svhn_small_conv",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("cifar10",),
            models=("devries", "confidnet"),
            backbones=("vgg13",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("cifar10",),
            models=("dg",),
            backbones=("vgg13",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("cifar100",),
            models=("devries", "confidnet"),
            backbones=("vgg13",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("cifar100",),
            models=("dg",),
            backbones=("vgg13",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 12, 15, 20),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("supercifar",),
            models=("devries", "confidnet"),
            backbones=("vgg13",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )


    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("supercifar",),
            models=("dg",),
            backbones=("vgg13",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("animals",),
            models=("devries", "confidnet"),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("animals",),
            models=("dg",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 15),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("camelyon",),
            models=("devries", "confidnet"),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("camelyon",),
            models=("dg",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("breeds",),
            models=("devries", "confidnet"),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(2),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("breeds",),
            models=("dg",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(2),
            rewards=(2.2, 3, 6, 10, 15),
            learning_rates=(None,)
        )
    )


    # precision study
    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/svhn_precision_study16"),
            datasets=("svhn",),
            models=("confidnet",),
            backbones=("svhn_small_conv",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/svhn_precision_study32"),
            datasets=("svhn",),
            models=("confidnet",),
            backbones=("svhn_small_conv",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/svhn_precision_study64"),
            datasets=("svhn",),
            models=("confidnet",),
            backbones=("svhn_small_conv",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/camelyon_precision_study16"),
            datasets=("camelyon",),
            models=("confidnet",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/camelyon_precision_study32"),
            datasets=("camelyon",),
            models=("confidnet",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/camelyon_precision_study64"),
            datasets=("camelyon",),
            models=("confidnet",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit_precision_study16"),
            datasets=("svhn",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit_precision_study32"),
            datasets=("svhn",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,)
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit_precision_study64"),
            datasets=("svhn",),
            models=("vit",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,)
        )
    )

    return _experiments

if __name__ == "__main__":
    pprint(get_all_experiments())