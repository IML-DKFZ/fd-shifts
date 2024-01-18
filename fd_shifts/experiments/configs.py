from pathlib import Path
from typing import Callable, Literal

from omegaconf import SI

from fd_shifts.configs import (
    ConfidMeasuresConfig,
    ConfidMetricsConfig,
    Config,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LRSchedulerConfig,
    Mode,
    ModelConfig,
    NetworkConfig,
    OptimizerConfig,
    OutputPathsConfig,
    OutputPathsPerMode,
    PerfMetricsConfig,
    QueryStudiesConfig,
    TestConfig,
    TrainerConfig,
    ValSplit,
)


def svhn_data_config(
    dataset: Literal["svhn", "svhn_openset"], img_size: int | tuple[int, int]
) -> DataConfig:
    augmentations = {
        "to_tensor": None,
        "resize": img_size,
        "normalize": [
            [0.4376821, 0.4437697, 0.47280442],
            [0.19803012, 0.20101562, 0.19703614],
        ],
    }

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return DataConfig(
        dataset="svhn"
        + ("_384" if img_size[0] == 384 else "")
        + ("_openset" if dataset == "svhn_openset" else ""),
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/svhn"),
        pin_memory=True,
        img_size=(img_size[0], img_size[1], 3),
        num_workers=12,
        num_classes=10,
        reproduce_confidnet_splits=True,
        augmentations={
            "train": augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        target_transforms=None,
        kwargs=None,
    )


def svhn_query_config(
    dataset: Literal["svhn", "svhn_openset"], img_size: int | tuple[int, int]
) -> QueryStudiesConfig:
    return QueryStudiesConfig(
        iid_study="svhn",
        noise_study=[],
        in_class_study=[],
        new_class_study=[
            cifar10_data_config(img_size),
            cifar100_data_config(img_size),
        ],  # , "tinyimagenet_384"],
    )


def cifar10_data_config(img_size: int | tuple[int, int]) -> DataConfig:
    augmentations = {
        "to_tensor": None,
        "resize": img_size,
        "normalize": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]],
    }

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return DataConfig(
        dataset="cifar10" + ("_384" if img_size[0] == 384 else ""),
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/cifar10"),
        pin_memory=True,
        img_size=(img_size[0], img_size[1], 3),
        num_workers=12,
        num_classes=10,
        reproduce_confidnet_splits=True,
        augmentations={
            "train": augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        target_transforms=None,
        kwargs=None,
    )


def cifar10_query_config(img_size: int | tuple[int, int]) -> QueryStudiesConfig:
    return QueryStudiesConfig(
        iid_study="cifar10",
        noise_study=[],
        in_class_study=[],
        new_class_study=[
            cifar100_data_config(img_size),
            svhn_data_config("svhn", img_size),
        ],  # , "tinyimagenet_384"],
    )


def cifar100_data_config(img_size: int | tuple[int, int]) -> DataConfig:
    augmentations = {
        "to_tensor": None,
        "resize": img_size,
        "normalize": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]],
    }

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return DataConfig(
        dataset="cifar100" + ("_384" if img_size[0] == 384 else ""),
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/cifar100"),
        pin_memory=True,
        img_size=(img_size[0], img_size[1], 3),
        num_workers=12,
        num_classes=100,
        reproduce_confidnet_splits=True,
        augmentations={
            "train": augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        target_transforms=None,
        kwargs=None,
    )


def wilds_animals_data_config(
    dataset: Literal["wilds_animals", "wilds_animals_ood_test"] = "wilds_animals",
    img_size: int | tuple[int, int] = 448,
) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augmentations = {
        "to_tensor": None,
        "resize": img_size,
        "normalize": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    }

    return DataConfig(
        dataset=dataset,
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/wilds_animals"),
        pin_memory=True,
        img_size=(img_size[0], img_size[1], 3),
        num_workers=8,
        num_classes=182,
        reproduce_confidnet_splits=False,
        augmentations={
            "train": augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        target_transforms=None,
        kwargs=None,
    )


def wilds_animals_query_config(img_size: int | tuple[int, int]) -> QueryStudiesConfig:
    return QueryStudiesConfig(
        iid_study="wilds_animals",
        noise_study=[],
        in_class_study=[wilds_animals_data_config("wilds_animals_ood_test", img_size)],
        new_class_study=[],
    )


def breeds_data_config(
    dataset: Literal["breeds", "breeds_ood_test"] = "breeds",
    img_size: int | tuple[int, int] = 224,
) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return DataConfig(
        dataset=dataset,
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/breeds"),
        img_size=(img_size[0], img_size[1], 3),
        num_classes=13,
        augmentations={
            "train": {
                "randomresized_crop": img_size,
                "hflip": True,
                "color_jitter": [0.1, 0.1, 0.1],
                "to_tensor": None,
                "normalize": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
            },
            "val": {
                "resize": 256 if img_size[0] == 224 else img_size,
                "center_crop": img_size,
                "to_tensor": None,
                "normalize": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
            },
            "test": {
                "resize": 256 if img_size[0] == 224 else img_size,
                "center_crop": img_size,
                "to_tensor": None,
                "normalize": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
            },
        },
        kwargs={"info_dir_path": "loaders/breeds_hierarchies"},
    )


def breeds_query_config(img_size: int | tuple[int, int]) -> QueryStudiesConfig:
    return QueryStudiesConfig(
        iid_study="breeds",
        noise_study=[],
        in_class_study=[breeds_data_config("breeds_ood_test", img_size)],
        new_class_study=[],
    )


__experiments: dict[str, Config] = {}


def svhn_modelvit_bbvit(lr: float, run: int, do: int, **kwargs) -> Config:
    return Config(
        exp=ExperimentConfig(
            group_name="vit",
            name=f"svhn_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
        ),
        pkgversion="0.0.1+f85760e",
        data=svhn_data_config("svhn", 384),
        trainer=TrainerConfig(
            num_epochs=None,
            num_steps=40000,
            batch_size=128,
            lr_scheduler=LRSchedulerConfig(
                init_args={
                    "class_path": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
                    "init_args": {
                        "warmup_epochs": 500,
                        "warmup_start_lr": 0,
                        "eta_min": 0,
                        "max_epochs": 40000,
                    },
                },
                class_path="fd_shifts.configs.LRSchedulerConfig",
            ),
            optimizer=OptimizerConfig(
                init_args={
                    "class_path": "torch.optim.SGD",
                    "init_args": {
                        "lr": 0.01,
                        "dampening": 0.0,
                        "momentum": 0.9,
                        "nesterov": False,
                        "maximize": False,
                        "weight_decay": 0.0,
                    },
                },
                class_path="fd_shifts.configs.OptimizerConfig",
            ),
            lr_scheduler_interval="epoch",
        ),
        model=ModelConfig(
            name="vit_model",
            network=NetworkConfig(
                name="vit",
            ),
            fc_dim=512,
            avg_pool=True,
            dropout_rate=0,
        ),
        eval=EvalConfig(
            val_tuning=True,
            query_studies=svhn_query_config("svhn", 384),
        ),
    )


def svhn_modeldg_bbvit(lr: float, run: int, do: int, rew: int | float) -> Config:
    config = svhn_modelvit_bbvit(lr=lr, run=run, do=do)
    config.trainer.num_steps = 60000
    config.trainer.lr_scheduler = LRSchedulerConfig(
        {
            "class_path": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
            "init_args": {
                "warmup_epochs": 500,
                "max_epochs": 60000,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0,
                "last_epoch": -1,
            },
        }
    )
    config.trainer.optimizer = OptimizerConfig(
        {
            "class_path": "torch.optim.SGD",
            "init_args": {
                "lr": lr,
                "dampening": 0.0,
                "momentum": 0.9,
                "nesterov": False,
                "maximize": False,
                "weight_decay": 0.0,
            },
        }
    )
    config.trainer.dg_pretrain_epochs = None
    config.trainer.dg_pretrain_steps = 20000
    config.trainer.lr_scheduler_interval = "step"
    config.exp.name = f"svhn_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}"
    config.model = ModelConfig(
        name="devries_model",
        network=NetworkConfig(
            name="vit",
            save_dg_backbone_path=Path("${exp.dir}/dg_backbone.ckpt"),
        ),
        fc_dim=768,
        avg_pool=True,
        dropout_rate=1,
        dg_reward=rew,
    )
    config.eval.ext_confid_name = "dg"
    config.eval.confidence_measures.test.append("ext")

    return config


def cifar10_modelvit_bbvit(lr: float, run: int, do: Literal[0, 1], **kwargs) -> Config:
    return Config(
        exp=ExperimentConfig(
            group_name="vit",
            name=f"cifar10_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
        ),
        data=cifar10_data_config(384),
        trainer=TrainerConfig(
            num_epochs=None,
            num_steps=40000,
            batch_size=128,
            lr_scheduler=LRSchedulerConfig(
                init_args={
                    "class_path": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
                    "init_args": {
                        "warmup_epochs": 500,
                        "warmup_start_lr": 0,
                        "eta_min": 0,
                        "max_epochs": 40000,
                    },
                },
                class_path="fd_shifts.configs.LRSchedulerConfig",
            ),
            optimizer=OptimizerConfig(
                init_args={
                    "class_path": "torch.optim.SGD",
                    "init_args": {
                        "lr": lr,
                        "dampening": 0.0,
                        "momentum": 0.9,
                        "nesterov": False,
                        "maximize": False,
                        "weight_decay": 0.0,
                    },
                },
                class_path="fd_shifts.configs.OptimizerConfig",
            ),
        ),
        model=ModelConfig(
            name="vit_model",
            network=NetworkConfig(
                name="vit",
            ),
            fc_dim=512,
            avg_pool=True,
            dropout_rate=do,
        ),
        eval=EvalConfig(
            query_studies=cifar10_query_config(384),
        ),
    )


def cifar10_modeldg_bbvit(
    lr: float, run: int, do: Literal[0, 1], rew: int | float
) -> Config:
    config = cifar10_modelvit_bbvit(lr=lr, run=run, do=do)
    config.trainer.num_steps = 60000
    config.trainer.lr_scheduler = LRSchedulerConfig(
        {
            "class_path": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
            "init_args": {
                "warmup_epochs": 500,
                "max_epochs": 60000,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0,
                "last_epoch": -1,
            },
        }
    )
    config.trainer.optimizer = OptimizerConfig(
        {
            "class_path": "torch.optim.SGD",
            "init_args": {
                "lr": lr,
                "dampening": 0.0,
                "momentum": 0.9,
                "nesterov": False,
                "maximize": False,
                "weight_decay": 0.0,
            },
        }
    )
    config.trainer.dg_pretrain_epochs = None
    config.trainer.dg_pretrain_steps = 20000
    config.trainer.lr_scheduler_interval = "step"
    config.exp.name = f"cifar10_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}"
    config.model = ModelConfig(
        name="devries_model",
        network=NetworkConfig(
            name="vit",
            save_dg_backbone_path=Path("${exp.dir}/dg_backbone.ckpt"),
        ),
        fc_dim=768,
        avg_pool=True,
        dropout_rate=1,
        dg_reward=rew,
    )
    config.eval.ext_confid_name = "dg"
    config.eval.confidence_measures.test.append("ext")

    return config


def clip(
    dataset: DataConfig,
    query_studies: QueryStudiesConfig,
    class_prefix: str | None = None,
    **kwargs,
):
    return Config(
        data=dataset,
        exp=ExperimentConfig(
            group_name="clip",
            name=f"{dataset.dataset}_modelclip_prefix{class_prefix.replace(' ', '-') if class_prefix else ''}",
        ),
        model=ModelConfig(
            name="clip_model",
            clip_class_prefix=class_prefix,
        ),
        eval=EvalConfig(
            query_studies=query_studies,
        ),
    )


def register(config_fn: Callable[..., Config], n_runs: int = 5, **kwargs):
    for run in range(n_runs):
        config = config_fn(**kwargs, run=run)
        __experiments[config.exp.name] = config


register(svhn_modelvit_bbvit, lr=0.03, do=1, rew=2.2)
register(svhn_modelvit_bbvit, lr=0.01, do=0, rew=2.2)
register(svhn_modelvit_bbvit, lr=0.01, do=1, rew=2.2)
register(svhn_modeldg_bbvit, lr=0.01, do=1, rew=2.2)
register(svhn_modeldg_bbvit, lr=0.01, do=1, rew=3)
register(svhn_modeldg_bbvit, lr=0.01, do=1, rew=6)
register(svhn_modeldg_bbvit, lr=0.01, do=1, rew=10)
register(svhn_modeldg_bbvit, lr=0.03, do=1, rew=2.2)
register(svhn_modeldg_bbvit, lr=0.03, do=1, rew=3)
register(svhn_modeldg_bbvit, lr=0.03, do=1, rew=6)
register(svhn_modeldg_bbvit, lr=0.03, do=1, rew=10)

register(cifar10_modelvit_bbvit, lr=3e-4, do=0, rew=2.2)
register(cifar10_modelvit_bbvit, lr=0.01, do=1, rew=2.2)
register(cifar10_modeldg_bbvit, lr=3e-4, do=0, rew=2.2)
register(cifar10_modeldg_bbvit, lr=0.01, do=1, rew=2.2)
register(cifar10_modeldg_bbvit, lr=3e-4, do=0, rew=3)
register(cifar10_modeldg_bbvit, lr=0.01, do=1, rew=3)
register(cifar10_modeldg_bbvit, lr=3e-4, do=0, rew=6)
register(cifar10_modeldg_bbvit, lr=0.01, do=1, rew=6)
register(cifar10_modeldg_bbvit, lr=3e-4, do=0, rew=10)
register(cifar10_modeldg_bbvit, lr=0.01, do=1, rew=10)

register(
    clip,
    n_runs=1,
    dataset=cifar10_data_config(img_size=224),
    query_studies=cifar10_query_config(224),
    class_prefix=None,
)
register(
    clip,
    n_runs=1,
    dataset=cifar10_data_config(img_size=224),
    query_studies=cifar10_query_config(224),
    class_prefix="a",
)
register(
    clip,
    n_runs=1,
    dataset=cifar10_data_config(img_size=224),
    query_studies=cifar10_query_config(224),
    class_prefix="a picture of a",
)
register(
    clip,
    n_runs=1,
    dataset=svhn_data_config("svhn", 224),
    query_studies=svhn_query_config("svhn", 224),
    class_prefix=None,
)
register(
    clip,
    n_runs=1,
    dataset=svhn_data_config("svhn", 224),
    query_studies=svhn_query_config("svhn", 224),
    class_prefix="a",
)
register(
    clip,
    n_runs=1,
    dataset=svhn_data_config("svhn", 224),
    query_studies=svhn_query_config("svhn", 224),
    class_prefix="a picture of a",
)
register(
    clip,
    n_runs=1,
    dataset=wilds_animals_data_config("wilds_animals", 224),
    query_studies=wilds_animals_query_config(224),
    class_prefix=None,
)
register(
    clip,
    n_runs=1,
    dataset=wilds_animals_data_config("wilds_animals", 224),
    query_studies=wilds_animals_query_config(224),
    class_prefix="a",
)
register(
    clip,
    n_runs=1,
    dataset=wilds_animals_data_config("wilds_animals", 224),
    query_studies=wilds_animals_query_config(224),
    class_prefix="a picture of a",
)
register(
    clip,
    n_runs=1,
    dataset=breeds_data_config("breeds", 224),
    query_studies=breeds_query_config(224),
    class_prefix=None,
)
register(
    clip,
    n_runs=1,
    dataset=breeds_data_config("breeds", 224),
    query_studies=breeds_query_config(224),
    class_prefix="a",
)
register(
    clip,
    n_runs=1,
    dataset=breeds_data_config("breeds", 224),
    query_studies=breeds_query_config(224),
    class_prefix="a picture of a",
)


def get_experiment_config(name: str) -> Config:
    return __experiments[name]


def list_experiment_configs() -> list[str]:
    return list(sorted(__experiments.keys()))
