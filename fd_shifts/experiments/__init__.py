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

    def overrides(self):
        match self.dataset:
            case "camelyon" | "animals" | "animals_openset":
                dataset = "wilds_" + self.dataset
            case "supercifar":
                dataset = "super_cifar100"
            case _:
                dataset = self.dataset

        match self.model:
            case "dg":
                model = "deepgamblers"
            case "vit":
                model = self.model
            case _:
                model = self.model

        if self.backbone == "vit":
            dataset = dataset + "_384"

        overrides = {
            "data": dataset + "_data",
            "study": model,
            "model.dropout_rate": self.dropout,
            "model.dg_reward": self.reward,
            "exp.group_name": str(self.to_path().parent.stem),
            "exp.name": str(self.to_path().name),
        }

        if self.learning_rate is not None:
            overrides["trainer.optimizer.lr"] = self.learning_rate

        if self.model == "confidnet":
            match dataset:
                case "breeds" | "breeds_384":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[300, 500]"'
                case "cifar10" | "cifar100" | "super_cifar100" | "cifar10_384" | "cifar100_384" | "super_cifar100_384":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[250, 450]"'
                case "svhn" | "svhn_openset" | "svhn_384" | "svhn_openset_384":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[100, 300]"'
                case "wilds_animals" | "wilds_animals_openset" | "wilds_animals_384" | "wilds_animals_openset_384":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[12, 17]"'
                case "wilds_camelyon" | "wilds_camelyon_384":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[5, 8]"'
                case _:
                    pass

        if self.backbone == "vit" and dataset in (
            "cifar100_384",
            "super_cifar100_384",
            "wilds_animals_384",
            "wilds_animals_openset_384",
        ):
            overrides["trainer.batch_size"] = 512
        elif self.backbone == "vit":
            overrides["trainer.batch_size"] = 128

        if self.backbone == "vit":
            match model:
                case "deepgamblers":
                    overrides["model.network.name"] = "vit"
                    overrides["model.fc_dim"] = 768
                case "devries":
                    overrides["model.network.backbone"] = "vit"
                case "confidnet":
                    overrides["model.network.backbone"] = "vit"
                    overrides["model.fc_dim"] = 768
                    overrides[
                        "trainer.callbacks.training_stages.pretrained_backbone_path"
                    ] = (
                        "${EXPERIMENT_ROOT_DIR%/}/"
                        + (
                            str(self.to_path())
                            .replace("modelconfidnet", "modelvit")
                            .replace("fd-shifts/", "")
                        )
                        + "/version_0/last.ckpt"
                    )
                case _:
                    pass

        return overrides

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


def get_all_experiments(
    with_hyperparameter_sweep=False, with_vit_special_runs=False
) -> list[Experiment]:
    _experiments = []

    # ViT Best lr runs
    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("svhn",),
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("svhn_openset",),
            models=("vit", "dg", "devries", "confidnet"),
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
            datasets=("svhn_openset",),
            models=("vit", "dg", "devries", "confidnet"),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("cifar10",),
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
            backbones=("vit",),
            learning_rates=(3e-3, 1e-2),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_animals_openset",),
            models=("vit", "dg", "devries", "confidnet"),
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
            datasets=("wilds_animals_openset",),
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            models=("vit", "dg", "devries", "confidnet"),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("svhn_openset",),
            models=("devries", "confidnet"),
            backbones=("svhn_small_conv",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("svhn_openset",),
            models=("dg",),
            backbones=("svhn_small_conv",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            rewards=(2.2, 3, 6, 10, 12, 15, 20),
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("animals_openset",),
            models=("devries", "confidnet"),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2,),
            learning_rates=(None,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("animals_openset",),
            models=("dg",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 15),
            learning_rates=(None,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("camelyon",),
            models=("devries", "confidnet"),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(10),
            rewards=(2.2,),
            learning_rates=(None,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts"),
            datasets=("camelyon",),
            models=("dg",),
            backbones=("resnet50",),
            dropouts=(0, 1),
            runs=range(10),
            rewards=(2.2, 3, 6, 10),
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            learning_rates=(None,),
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
            rewards=(0,),
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
            rewards=(0,),
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
            rewards=(0,),
        )
    )

    if not with_vit_special_runs:
        _experiments = list(
            filter(
                lambda exp: not (exp.backbone == "vit" and exp.model != "vit"), _experiments
            )
        )

    return _experiments


if __name__ == "__main__":
    pprint(get_all_experiments())
