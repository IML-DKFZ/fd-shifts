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

        if dataset in ("cifar10", "cifar100", "supercifar") and self.dropout:
            overrides["model.avg_pool"] = False

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

        if "medshifts" in str(self.group_dir):
            if self.model == "deepgamblers":
                overrides["model.network.name"] = self.backbone

            if "but" in self.dataset:
                overrides[
                    "eval.query_studies.in_class_study"
                ] = f'"[{self.dataset.replace("but", "")}]"'

            if self.dataset.startswith("dermoscopyall"):
                overrides["data"] = "dermoscopyall_data"
                overrides["data.dataset"] = self.dataset
                if self.model == "confidnet":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[20, 25]"'

                if self.dataset == "dermoscopyallham10000subclass":
                    overrides["data.dataset"] = "dermoscopyallham10000subbig"
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = f'"[dermoscopyallham10000subsmall]"'

                if self.dataset == "dermoscopyallham10000multi":
                    overrides["data.num_classes"] = 7
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = f'"[dermoscopyallham10000multi]"'

                if self.dataset == "dermoscopyall":
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = '"[dermoscopyallcorrbrhigh, dermoscopyallcorrbrhighhigh, dermoscopyallcorrbrlow, dermoscopyallcorrbrlowlow, dermoscopyallcorrgaunoilow, dermoscopyallcorrgaunoilowlow]"'

            if self.dataset.startswith("lidc_idriall"):
                overrides["data"] = "lidc_idriall_data"
                overrides["data.dataset"] = self.dataset
                if self.dataset != "lidc_idriall":
                    overrides["data.dataset"] += "_iid"
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = f'"[{self.dataset + "_ood"}]"'
                else:
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = '"[lidc_idriallcorrbrhigh, lidc_idriallcorrbrhighhigh, lidc_idriallcorrbrlow, lidc_idriallcorrbrlowlow, lidc_idriallcorrgaunoilow, lidc_idriallcorrgaunoilowlow]"'

                if self.model == "confidnet":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[45, 60]"'

            if self.dataset.startswith("rxrx1all"):
                overrides["data"] = "rxrx1all_data"
                overrides["data.dataset"] = self.dataset

                if self.dataset == "rxrx1all":
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = '"[rxrx1allcorrbrhigh, rxrx1allcorrbrhighhigh, rxrx1allcorrbrlow, rxrx1allcorrbrlowlow, rxrx1allcorrgaunoilow, rxrx1allcorrgaunoilowlow]"'
                else:
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = f'"[{self.dataset.replace("large", "small")}]"'

                if self.model == "confidnet":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[120, 150]"'

            if self.dataset.startswith("xray_chestall"):
                overrides["data"] = "xray_chestall_data"
                overrides["data.dataset"] = self.dataset

                if self.dataset == "xray_chestall":
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = '"[xray_chestallcorrbrhigh, xray_chestallcorrbrhighhigh, xray_chestallcorrbrlow, xray_chestallcorrbrlowlow, xray_chestallcorrgaunoilow, xray_chestallcorrgaunoilowlow, xray_chestallcorrletter]"'
                else:
                    overrides[
                        "eval.query_studies.in_class_study"
                    ] = f'"[{self.dataset.replace("but", "")}]"'

                if self.model == "confidnet":
                    overrides[
                        "trainer.callbacks.training_stages.milestones"
                    ] = '"[40, 50]"'
                else:
                    overrides["trainer.optimizer.weight_decay"] = 1e-5

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

        if "medshifts" in str(self.group_dir.stem):
            return self.group_dir / (
                f"ms_{self.dataset}/"
                f"{self.model}_"
                f"bb{self.backbone}_"
                f"run{self.run + 1}"
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


def get_ms_experiments() -> list[Experiment]:
    _experiments = []

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "dermoscopyall",
                "dermoscopyallbutbarcelona",
                "dermoscopyallbutmskcc",
                "dermoscopyallham10000multi",
                "dermoscopyallham10000subclass",
            ),
            models=("deepgamblers", "devries", "confidnet"),
            backbones=("efficientnetb4",),
            dropouts=(1,),
            runs=range(3),
            rewards=(10,),
            learning_rates=(3e-5,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "dermoscopyall",
                "dermoscopyallbutbarcelona",
                "dermoscopyallbutmskcc",
                "dermoscopyallham10000multi",
                "dermoscopyallham10000subclass",
            ),
            models=("deepgamblers", "vit"),
            backbones=("vit",),
            dropouts=(1,),
            runs=range(3),
            rewards=(10,),
            learning_rates=(3e-5,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "lidc_idriall",
                "lidc_idriall_calcification",
                "lidc_idriall_spiculation",
                "lidc_idriall_texture",
            ),
            models=("deepgamblers", "devries", "confidnet"),
            backbones=("densenet121",),
            dropouts=(1,),
            runs=range(3),
            rewards=(20,),
            learning_rates=(1.5e-4,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "lidc_idriall",
                "lidc_idriall_calcification",
                "lidc_idriall_spiculation",
                "lidc_idriall_texture",
            ),
            models=("deepgamblers", "vit"),
            backbones=("vit",),
            dropouts=(1,),
            runs=range(3),
            rewards=(20,),
            learning_rates=(1.5e-4,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "rxrx1all_large_set1",
                "rxrx1all_large_set2",
                "rxrx1all",
            ),
            models=("deepgamblers", "devries", "confidnet"),
            backbones=("densenet161",),
            dropouts=(1,),
            runs=range(3),
            rewards=(10,),
            learning_rates=(1.5e-4,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "rxrx1all_large_set1",
                "rxrx1all_large_set2",
                "rxrx1all",
            ),
            models=("deepgamblers", "vit"),
            backbones=("vit",),
            dropouts=(1,),
            runs=range(3),
            rewards=(10,),
            learning_rates=(1.5e-4,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "xray_chestallbutchexpert",
                "xray_chestallbutnih14",
                "xray_chestall",
            ),
            models=("deepgamblers", "devries", "confidnet"),
            backbones=("densenet121",),
            dropouts=(1,),
            runs=range(3),
            rewards=(10,),
            learning_rates=(5e-4,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("medshifts/"),
            datasets=(
                "xray_chestallbutchexpert",
                "xray_chestallbutnih14",
                "xray_chestall",
            ),
            models=("deepgamblers", "vit"),
            backbones=("vit",),
            dropouts=(1,),
            runs=range(3),
            rewards=(10,),
            learning_rates=(5e-4,),
        )
    )
    return _experiments


def get_all_experiments(
    with_hyperparameter_sweep=False, with_vit_special_runs=True, with_ms_runs=True
) -> list[Experiment]:
    _experiments = []

    # ViT Best lr runs
    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("svhn",),
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            datasets=("svhn", "svhn_openset"),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(3e-2,),
            dropouts=(1,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("svhn", "svhn_openset"),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("svhn_openset",),
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            datasets=("cifar10",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(1,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("cifar10",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(3e-4,),
            dropouts=(0,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("cifar100",),
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            datasets=("cifar100",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(1, 0),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 12, 15, 20),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("super_cifar100",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(3e-3,),
            dropouts=(0,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 12, 15, 20),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("super_cifar100",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(1e-3,),
            dropouts=(1,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 12, 15, 20),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_animals",),
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            datasets=(
                "wilds_animals",
                "wilds_animals_openset",
            ),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(1e-3,),
            dropouts=(0,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 15),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=(
                "wilds_animals",
                "wilds_animals_openset",
            ),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(3e-3,),
            dropouts=(0, 1),
            runs=range(5),
            rewards=(2.2, 3, 6, 10, 15),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_camelyon",),
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
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
            datasets=("wilds_camelyon",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(1e-3,),
            dropouts=(0,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("wilds_camelyon",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(3e-3,),
            dropouts=(1,),
            runs=range(5),
            rewards=(2.2, 3, 6, 10),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("breeds",),
            models=("vit", "devries", "confidnet"),
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
            models=("vit", "devries", "confidnet"),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(1,),
            runs=range(2),
            rewards=(0,),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("breeds",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(3e-3, 1e-3),
            dropouts=(0,),
            runs=range(2),
            rewards=(2.2, 3, 6, 10, 15),
        )
    )

    _experiments.extend(
        Experiment.from_iterables(
            group_dir=Path("fd-shifts/vit"),
            datasets=("breeds",),
            models=("dg",),
            backbones=("vit",),
            learning_rates=(1e-2,),
            dropouts=(1,),
            runs=range(2),
            rewards=(2.2, 3, 6, 10, 15),
        )
    )
    # ViT Best lr runs

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
                lambda exp: not (exp.backbone == "vit" and exp.model != "vit"),
                _experiments,
            )
        )

    # if with_ms_runs:
    _experiments.extend(get_ms_experiments())

    return _experiments


if __name__ == "__main__":
    pprint(get_all_experiments())
