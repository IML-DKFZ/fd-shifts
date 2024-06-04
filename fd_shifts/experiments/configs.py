from collections.abc import Callable
from copy import deepcopy
from typing import Literal

from omegaconf import SI

from fd_shifts.configs import (
    Config,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    QueryStudiesConfig,
)

DEFAULT_VIT_INFERENCE_IMG_SIZE = 384
DEFAULT_VIT_PRETRAIN_IMG_SIZE = 224
DEFAULT_SMALL_IMG_SIZE = 32
DEFAULT_CAMELYON_IMG_SIZE = 96


def svhn_data_config(
    dataset: Literal["svhn", "svhn_openset"], img_size: int | tuple[int, int] = 32
) -> DataConfig:
    augmentations = {
        "to_tensor": None,
        "normalize": [
            [0.4376821, 0.4437697, 0.47280442],
            [0.19803012, 0.20101562, 0.19703614],
        ],
    }

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    if img_size[0] != DEFAULT_SMALL_IMG_SIZE:
        augmentations["resize"] = img_size[0]

    return DataConfig(
        dataset="svhn" + ("_openset" if dataset == "svhn_openset" else ""),
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
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return QueryStudiesConfig(
        iid_study="svhn"
        + ("_384" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else ""),
        new_class_study=[
            cifar10_data_config(img_size=img_size),
            cifar100_data_config(img_size=img_size),
            tinyimagenet_data_config(img_size),
        ],
    )


def cifar10_data_config(
    dataset: Literal["cifar10", "corrupt_cifar10"] = "cifar10",
    img_size: int | tuple[int, int] = DEFAULT_SMALL_IMG_SIZE,
) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augmentations = {
        "to_tensor": None,
        "normalize": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]],
    }
    if img_size[0] != DEFAULT_SMALL_IMG_SIZE:
        augmentations["resize"] = img_size[0]

    train_augmentations = deepcopy(augmentations)

    if img_size[0] != DEFAULT_VIT_INFERENCE_IMG_SIZE:
        train_augmentations["random_crop"] = [DEFAULT_SMALL_IMG_SIZE, 4]
        train_augmentations["hflip"] = True
        if dataset == "corrupt_cifar10":
            train_augmentations["rotate"] = 15
        else:
            train_augmentations["cutout"] = 16

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return DataConfig(
        dataset=dataset,
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/" + dataset),
        pin_memory=True,
        img_size=(img_size[0], img_size[1], 3),
        num_workers=12,
        num_classes=10,
        reproduce_confidnet_splits=True,
        augmentations={
            "train": train_augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        target_transforms=None,
        kwargs=None,
    )


def cifar10_query_config(img_size: int | tuple[int, int]) -> QueryStudiesConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return QueryStudiesConfig(
        iid_study="cifar10"
        + ("_384" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else ""),
        noise_study=cifar10_data_config("corrupt_cifar10", img_size),
        new_class_study=[
            cifar100_data_config(img_size=img_size),
            svhn_data_config("svhn", img_size),
            tinyimagenet_data_config(img_size),
        ],
    )


def cifar100_data_config(
    dataset: Literal["cifar100", "corrupt_cifar100", "super_cifar100"] = "cifar100",
    img_size: int | tuple[int, int] = DEFAULT_SMALL_IMG_SIZE,
) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augmentations = {
        "to_tensor": None,
        "normalize": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]],
    }
    if img_size[0] != DEFAULT_SMALL_IMG_SIZE:
        augmentations["resize"] = img_size[0]

    train_augmentations = deepcopy(augmentations)

    if img_size[0] != DEFAULT_VIT_INFERENCE_IMG_SIZE:
        train_augmentations["random_crop"] = [DEFAULT_SMALL_IMG_SIZE, 4]
        train_augmentations["hflip"] = True
        if dataset == "corrupt_cifar100":
            train_augmentations["rotate"] = 15
        else:
            train_augmentations["cutout"] = 16

    return DataConfig(
        dataset=dataset,
        data_dir=SI(
            "${oc.env:DATASET_ROOT_DIR}/"
            + ("cifar100" if dataset in ["cifar100", "super_cifar100"] else dataset)
        ),
        pin_memory=True,
        img_size=(img_size[0], img_size[1], 3),
        num_workers=12,
        num_classes=19 if dataset == "super_cifar100" else 100,
        reproduce_confidnet_splits=True,
        augmentations={
            "train": train_augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        target_transforms=None,
        kwargs=None,
    )


def cifar100_query_config(
    img_size: int | tuple[int, int],
    dataset: Literal["cifar100", "super_cifar100"] = "cifar100",
) -> QueryStudiesConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return QueryStudiesConfig(
        iid_study=dataset
        + ("_384" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else ""),
        noise_study=(
            cifar100_data_config("corrupt_cifar100", img_size)
            if dataset == "cifar100"
            else DataConfig()
        ),
        in_class_study=[],
        new_class_study=(
            [
                cifar10_data_config(img_size=img_size),
                svhn_data_config("svhn", img_size),
                tinyimagenet_data_config(img_size),
            ]
            if dataset == "cifar100"
            else []
        ),
    )


def wilds_animals_data_config(
    dataset: Literal["wilds_animals", "wilds_animals_ood_test"] = "wilds_animals",
    img_size: int | tuple[int, int] = 448,
) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augmentations = {
        "to_tensor": None,
        "resize": (
            img_size[0] if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else img_size
        ),
        "normalize": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    }

    if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE:
        augmentations["center_crop"] = 384

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


def wilds_animals_query_config(
    img_size: int | tuple[int, int] = 448
) -> QueryStudiesConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return QueryStudiesConfig(
        iid_study="wilds_animals"
        + ("_384" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else ""),
        in_class_study=[wilds_animals_data_config("wilds_animals_ood_test", img_size)],
        new_class_study=[],
    )


def wilds_camelyon_data_config(
    dataset: Literal["wilds_camelyon", "wilds_camelyon_ood_test"] = "wilds_camelyon",
    img_size: int | tuple[int, int] = DEFAULT_CAMELYON_IMG_SIZE,
) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augmentations = {
        "to_tensor": None,
        "normalize": [
            [0.485, 0.456, 0.406],
            [
                0.229,
                0.384 if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else 0.224,
                0.225,
            ],
        ],
    }

    if img_size[0] != DEFAULT_CAMELYON_IMG_SIZE:
        augmentations["resize"] = img_size[0]

    return DataConfig(
        dataset=dataset,
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/wilds_camelyon"),
        pin_memory=True,
        img_size=(img_size[0], img_size[1], 3),
        num_workers=8,
        num_classes=2,
        reproduce_confidnet_splits=False,
        augmentations={
            "train": augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        target_transforms=None,
        kwargs=None,
    )


def wilds_camelyon_query_config(
    img_size: int | tuple[int, int] = 96
) -> QueryStudiesConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return QueryStudiesConfig(
        iid_study="wilds_camelyon"
        + ("_384" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else ""),
        in_class_study=[
            wilds_camelyon_data_config("wilds_camelyon_ood_test", img_size)
        ],
        new_class_study=[],
    )


def breeds_data_config(
    dataset: Literal["breeds", "breeds_ood_test"] = "breeds",
    img_size: int | tuple[int, int] = DEFAULT_VIT_PRETRAIN_IMG_SIZE,
) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augmentations = {
        "resize": 256 if img_size[0] == DEFAULT_VIT_PRETRAIN_IMG_SIZE else img_size[0],
        "center_crop": img_size[0],
        "to_tensor": None,
        "normalize": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    }

    train_augmentations = deepcopy(augmentations)

    if img_size[0] != DEFAULT_VIT_INFERENCE_IMG_SIZE:
        train_augmentations["randomresized_crop"] = img_size[0]
        train_augmentations["hflip"] = True
        train_augmentations["color_jitter"] = [0.1, 0.1, 0.1]
        del train_augmentations["resize"]
        del train_augmentations["center_crop"]

    return DataConfig(
        dataset=dataset,
        data_dir=SI("${oc.env:DATASET_ROOT_DIR}/breeds"),
        img_size=(img_size[0], img_size[1], 3),
        num_classes=13,
        augmentations={
            "train": train_augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        kwargs={"info_dir_path": "loaders/breeds_hierarchies"},
    )


def breeds_query_config(
    img_size: int | tuple[int, int] = DEFAULT_VIT_PRETRAIN_IMG_SIZE
) -> QueryStudiesConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return QueryStudiesConfig(
        iid_study="breeds"
        + ("_384" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else ""),
        in_class_study=[breeds_data_config("breeds_ood_test", img_size)],
    )


def tinyimagenet_data_config(img_size: int | tuple[int, int] = 64) -> DataConfig:
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augmentations = {
        "to_tensor": None,
        "normalize": [[0.485, 0.456, 0.406], [0.229, 0.384, 0.225]],
    }

    if img_size[0] != 64:  # noqa: PLR2004
        augmentations["resize"] = img_size

    return DataConfig(
        dataset="tinyimagenet"
        + ("_384" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else "_resize"),
        data_dir=SI(
            "${oc.env:DATASET_ROOT_DIR}/tinyimagenet"
            + ("" if img_size[0] == DEFAULT_VIT_INFERENCE_IMG_SIZE else "_resize")
        ),
        img_size=(img_size[0], img_size[1], 3),
        num_classes=200,
        augmentations={
            "train": augmentations,
            "val": augmentations,
            "test": augmentations,
        },
        kwargs={},
    )


__dataset_configs: dict[str, DataConfig] = {
    "svhn": svhn_data_config("svhn"),
    "svhn_384": svhn_data_config("svhn", DEFAULT_VIT_INFERENCE_IMG_SIZE),
    "cifar10": cifar10_data_config(),
    "cifar10_384": cifar10_data_config(img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE),
    "cifar100": cifar100_data_config(),
    "cifar100_384": cifar100_data_config(img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE),
    "super_cifar100": cifar100_data_config(dataset="super_cifar100"),
    "super_cifar100_384": cifar100_data_config(
        img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE, dataset="super_cifar100"
    ),
    "corrupt_cifar10": cifar10_data_config(dataset="corrupt_cifar10"),
    "corrupt_cifar10_384": cifar10_data_config(
        dataset="corrupt_cifar10", img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE
    ),
    "corrupt_cifar100": cifar100_data_config(dataset="corrupt_cifar100"),
    "corrupt_cifar100_384": cifar100_data_config(
        dataset="corrupt_cifar100", img_size=384
    ),
    "wilds_animals": wilds_animals_data_config(),
    "wilds_animals_ood_test": wilds_animals_data_config("wilds_animals_ood_test"),
    "wilds_animals_ood_test_384": wilds_animals_data_config(
        "wilds_animals_ood_test", 384
    ),
    "wilds_camelyon": wilds_camelyon_data_config(),
    "wilds_camelyon_ood_test": wilds_camelyon_data_config("wilds_camelyon_ood_test"),
    "wilds_camelyon_ood_test_384": wilds_camelyon_data_config(
        "wilds_camelyon_ood_test", 384
    ),
    "breeds": breeds_data_config(),
    "breeds_ood_test": breeds_data_config("breeds_ood_test"),
    "breeds_ood_test_384": breeds_data_config(
        "breeds_ood_test", DEFAULT_VIT_INFERENCE_IMG_SIZE
    ),
    "tinyimagenet_384": tinyimagenet_data_config(DEFAULT_VIT_INFERENCE_IMG_SIZE),
    "tinyimagenet_resize": tinyimagenet_data_config(DEFAULT_SMALL_IMG_SIZE),
}

__query_configs: dict[str, QueryStudiesConfig] = {
    "svhn": svhn_query_config("svhn", DEFAULT_SMALL_IMG_SIZE),
    "cifar10": cifar10_query_config(DEFAULT_SMALL_IMG_SIZE),
    "cifar100": cifar100_query_config(DEFAULT_SMALL_IMG_SIZE),
    "super_cifar100": cifar100_query_config(DEFAULT_SMALL_IMG_SIZE, "super_cifar100"),
    "wilds_animals": wilds_animals_query_config(),
    "wilds_camelyon": wilds_camelyon_query_config(),
    "breeds": breeds_query_config(),
}


def get_dataset_config(name: str) -> DataConfig:
    return __dataset_configs[name]


def get_query_config(name: str) -> QueryStudiesConfig:
    return __query_configs[name]


__experiments: dict[str, Config] = {}


def cnn(group_name: str, name: str) -> Config:
    config = Config(exp=ExperimentConfig(group_name=group_name, name=name))
    config.trainer.batch_size = 128
    config.trainer.lr_scheduler = LRSchedulerConfig(
        init_args={
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {},
        },
        class_path="fd_shifts.configs.LRSchedulerConfig",
    )
    config.trainer.optimizer = OptimizerConfig(
        init_args={
            "class_path": "torch.optim.SGD",
            "init_args": {
                "dampening": 0.0,
                "momentum": 0.9,
                "nesterov": False,
                "maximize": False,
                "weight_decay": 0.0,
            },
        },
        class_path="fd_shifts.configs.OptimizerConfig",
    )
    config.model.confidnet_fc_dim = 400
    return config


def cnn_animals(name: str) -> Config:
    config = cnn("animals_paper_sweep", name=name)
    config.data = wilds_animals_data_config()
    config.trainer.optimizer.init_args["init_args"]["lr"] = 0.001  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.fc_dim = 2048
    config.model.avg_pool = True
    config.eval.query_studies = wilds_animals_query_config()
    return config


def cnn_animals_modelconfidnet(run: int, do: int, **kwargs) -> Config:
    config = cnn_animals(name=f"confidnet_bbresnet50_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 20
    config.trainer.num_epochs_backbone = 12
    config.trainer.learning_rate_confidnet = 0.0001
    config.trainer.learning_rate_confidnet_finetune = 1e-06
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 12  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.callbacks["training_stages"] = {}
    config.trainer.callbacks["training_stages"]["milestones"] = [12, 17]
    config.trainer.callbacks["training_stages"]["pretrained_backbone_path"] = None
    config.trainer.callbacks["training_stages"]["pretrained_confidnet_path"] = None
    config.trainer.callbacks["training_stages"]["disable_dropout_at_finetuning"] = True
    config.trainer.callbacks["training_stages"]["confidnet_lr_scheduler"] = False
    config.model.name = "confidnet_model"
    config.model.dropout_rate = do
    config.model.network.name = "confidnet_and_enc"
    config.model.network.backbone = "resnet50"
    config.eval.ext_confid_name = "tcp"
    return config


def cnn_animals_modeldevries(run: int, do: int, **kwargs) -> Config:
    config = cnn_animals(name=f"devries_bbresnet50_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 12
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 12  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = True  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dg_reward = -1
    config.model.dropout_rate = do
    config.model.network.name = "devries_and_enc"
    config.model.network.backbone = "resnet50"
    config.eval.ext_confid_name = "devries"
    return config


def cnn_animals_modeldg(run: int, do: int, rew: float) -> Config:
    config = cnn_animals(name=f"dg_bbresnet50_do{do}_run{run + 1}_rew{rew}")
    config.trainer.num_epochs = 18
    config.trainer.dg_pretrain_epochs = 6
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 18  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = False  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dropout_rate = do
    config.model.dg_reward = rew
    config.model.network.name = "resnet50"
    config.eval.ext_confid_name = "dg"
    return config


def cnn_camelyon(name: str) -> Config:
    config = cnn("camelyon_paper_sweep", name=name)
    config.data = wilds_camelyon_data_config()
    config.trainer.optimizer.init_args["init_args"]["lr"] = 0.01  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["weight_decay"] = 0.01  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.fc_dim = 2048
    config.model.avg_pool = True
    config.eval.query_studies = wilds_camelyon_query_config()
    return config


def cnn_camelyon_modelconfidnet(run: int, do: int, **kwargs) -> Config:
    config = cnn_camelyon(f"confidnet_bbresnet50_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 9
    config.trainer.num_epochs_backbone = 5
    config.trainer.learning_rate_confidnet = 0.0001
    config.trainer.learning_rate_confidnet_finetune = 1e-06
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 5  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.callbacks["training_stages"] = {}
    config.trainer.callbacks["training_stages"]["milestones"] = [5, 8]
    config.trainer.callbacks["training_stages"]["pretrained_backbone_path"] = None
    config.trainer.callbacks["training_stages"]["pretrained_confidnet_path"] = None
    config.trainer.callbacks["training_stages"]["disable_dropout_at_finetuning"] = True
    config.trainer.callbacks["training_stages"]["confidnet_lr_scheduler"] = False
    config.model.name = "confidnet_model"
    config.model.dropout_rate = do
    config.model.network.name = "confidnet_and_enc"
    config.model.network.backbone = "resnet50"
    config.eval.ext_confid_name = "tcp"
    return config


def cnn_camelyon_modeldevries(run: int, do: int, **kwargs) -> Config:
    config = cnn_camelyon(f"devries_bbresnet50_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 5
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 5  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = True  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dropout_rate = do
    config.model.dg_reward = -1
    config.model.network.name = "devries_and_enc"
    config.model.network.backbone = "resnet50"
    config.eval.ext_confid_name = "devries"
    return config


def cnn_camelyon_modeldg(run: int, do: int, rew: float) -> Config:
    config = cnn_camelyon(f"dg_bbresnet50_do{do}_run{run + 1}_rew{rew}")
    config.trainer.num_epochs = 8
    config.trainer.dg_pretrain_epochs = 3
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 8  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = False  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dg_reward = rew
    config.model.dropout_rate = do
    config.model.network.name = "resnet50"
    config.eval.ext_confid_name = "dg"
    return config


def cnn_svhn(name: str) -> Config:
    config = cnn("svhn_paper_sweep", name=name)
    config.data = svhn_data_config("svhn", img_size=DEFAULT_SMALL_IMG_SIZE)
    config.trainer.optimizer.init_args["init_args"]["lr"] = 0.01  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["weight_decay"] = 0.0005  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.fc_dim = 512
    config.model.avg_pool = True
    config.eval.query_studies = svhn_query_config(
        "svhn", img_size=DEFAULT_SMALL_IMG_SIZE
    )
    return config


def cnn_svhn_modelconfidnet(run: int, do: int, **kwargs) -> Config:
    config = cnn_svhn(f"confidnet_bbsvhn_small_conv_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 320
    config.trainer.num_epochs_backbone = 100
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 100  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.learning_rate_confidnet = 0.0001
    config.trainer.learning_rate_confidnet_finetune = 1e-06
    config.trainer.callbacks["training_stages"] = {}
    config.trainer.callbacks["training_stages"]["milestones"] = [100, 300]
    config.trainer.callbacks["training_stages"]["pretrained_backbone_path"] = None
    config.trainer.callbacks["training_stages"]["pretrained_confidnet_path"] = None
    config.trainer.callbacks["training_stages"]["disable_dropout_at_finetuning"] = True
    config.trainer.callbacks["training_stages"]["confidnet_lr_scheduler"] = False
    config.model.name = "confidnet_model"
    config.model.dropout_rate = do
    config.model.network.name = "confidnet_and_enc"
    config.model.network.backbone = "svhn_small_conv"
    config.eval.ext_confid_name = "tcp"
    return config


def cnn_svhn_modeldevries(run: int, do: int, **kwargs) -> Config:
    config = cnn_svhn(f"devries_bbsvhn_small_conv_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 100
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 100  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = True  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dropout_rate = do
    config.model.dg_reward = -1
    config.model.network.name = "devries_and_enc"
    config.model.network.backbone = "svhn_small_conv"
    config.eval.ext_confid_name = "devries"
    return config


def cnn_svhn_modeldg(run: int, do: int, rew: float) -> Config:
    config = cnn_svhn(f"dg_bbsvhn_small_conv_do{do}_run{run + 1}_rew{rew}")
    config.trainer.num_epochs = 150
    config.trainer.dg_pretrain_epochs = 50
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 150  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = False  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dg_reward = rew
    config.model.dropout_rate = do
    config.model.network.name = "svhn_small_conv"
    config.eval.ext_confid_name = "dg"
    return config


def cnn_cifar10(name: str) -> Config:
    config = cnn("cifar10_paper_sweep", name=name)
    config.data = cifar10_data_config(img_size=DEFAULT_SMALL_IMG_SIZE)
    config.trainer.optimizer.init_args["init_args"]["lr"] = 0.1  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["weight_decay"] = 0.0005  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.fc_dim = 512
    config.eval.query_studies = cifar10_query_config(img_size=DEFAULT_SMALL_IMG_SIZE)
    return config


def cnn_cifar10_modelconfidnet(run: int, do: int, **kwargs) -> Config:
    config = cnn_cifar10(f"confidnet_bbvgg13_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 470
    config.trainer.num_epochs_backbone = 250
    config.trainer.learning_rate_confidnet = 0.0001
    config.trainer.learning_rate_confidnet_finetune = 1e-06
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 250  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.callbacks["training_stages"] = {}
    config.trainer.callbacks["training_stages"]["milestones"] = [250, 450]
    config.trainer.callbacks["training_stages"]["pretrained_backbone_path"] = None
    config.trainer.callbacks["training_stages"]["pretrained_confidnet_path"] = None
    config.trainer.callbacks["training_stages"]["disable_dropout_at_finetuning"] = True
    config.trainer.callbacks["training_stages"]["confidnet_lr_scheduler"] = False
    config.model.name = "confidnet_model"
    config.model.dropout_rate = do
    config.model.avg_pool = do == 0
    config.model.network.name = "confidnet_and_enc"
    config.model.network.backbone = "vgg13"
    config.eval.ext_confid_name = "tcp"
    return config


def cnn_cifar10_modeldevries(run: int, do: int, **kwargs) -> Config:
    config = cnn_cifar10(f"devries_bbvgg13_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 250
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 250  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = True  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dropout_rate = do
    config.model.avg_pool = do == 0
    config.model.dg_reward = -1
    config.model.network.name = "devries_and_enc"
    config.model.network.backbone = "vgg13"
    config.eval.ext_confid_name = "devries"
    return config


def cnn_cifar10_modeldg(run: int, do: int, rew: float) -> Config:
    config = cnn_cifar10(f"dg_bbvgg13_do{do}_run{run + 1}_rew{rew}")
    config.trainer.num_epochs = 300
    config.trainer.dg_pretrain_epochs = 100
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 300  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = False  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dg_reward = rew
    config.model.dropout_rate = do
    config.model.avg_pool = do == 0
    config.model.network.name = "vgg13"
    config.eval.ext_confid_name = "dg"
    return config


def cnn_cifar100(name: str) -> Config:
    config = cnn("cifar100_paper_sweep", name=name)
    config.data = cifar100_data_config(img_size=DEFAULT_SMALL_IMG_SIZE)
    config.trainer.optimizer.init_args["init_args"]["lr"] = 0.1  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["weight_decay"] = 0.0005  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.fc_dim = 512
    config.eval.query_studies = cifar100_query_config(img_size=DEFAULT_SMALL_IMG_SIZE)
    return config


def cnn_cifar100_modelconfidnet(run: int, do: int, **kwargs) -> Config:
    config = cnn_cifar100(f"confidnet_bbvgg13_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 470
    config.trainer.num_epochs_backbone = 250
    config.trainer.learning_rate_confidnet = 0.0001
    config.trainer.learning_rate_confidnet_finetune = 1e-06
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 250  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.callbacks["training_stages"] = {}
    config.trainer.callbacks["training_stages"]["milestones"] = [250, 450]
    config.trainer.callbacks["training_stages"]["pretrained_backbone_path"] = None
    config.trainer.callbacks["training_stages"]["pretrained_confidnet_path"] = None
    config.trainer.callbacks["training_stages"]["disable_dropout_at_finetuning"] = True
    config.trainer.callbacks["training_stages"]["confidnet_lr_scheduler"] = False
    config.model.name = "confidnet_model"
    config.model.avg_pool = do == 0
    config.model.dropout_rate = do
    config.model.network.name = "confidnet_and_enc"
    config.model.network.backbone = "vgg13"
    config.eval.ext_confid_name = "tcp"
    return config


def cnn_cifar100_modeldevries(run: int, do: int, **kwargs) -> Config:
    config = cnn_cifar100(f"devries_bbvgg13_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 250
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 250  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = True  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dropout_rate = do
    config.model.dg_reward = -1
    config.model.avg_pool = do == 0
    config.model.network.name = "devries_and_enc"
    config.model.network.backbone = "vgg13"
    config.eval.ext_confid_name = "devries"
    return config


def cnn_cifar100_modeldg(run: int, do: int, rew: float) -> Config:
    config = cnn_cifar100(f"dg_bbvgg13_do{do}_run{run + 1}_rew{rew}")
    config.trainer.num_epochs = 300
    config.trainer.dg_pretrain_epochs = 100
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 300  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = False  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dg_reward = rew
    config.model.dropout_rate = do
    config.model.avg_pool = do == 0
    config.model.network.name = "vgg13"
    config.eval.ext_confid_name = "dg"
    return config


def cnn_super_cifar100_modelconfidnet(run: int, do: int, **kwargs) -> Config:
    config = cnn_cifar100_modelconfidnet(run, do, **kwargs)
    config.exp.group_name = "supercifar_paper_sweep"
    config.data = cifar100_data_config(
        dataset="super_cifar100", img_size=DEFAULT_SMALL_IMG_SIZE
    )
    config.eval.query_studies = cifar100_query_config(
        dataset="super_cifar100", img_size=32
    )
    return config


def cnn_super_cifar100_modeldevries(run: int, do: int, **kwargs) -> Config:
    config = cnn_cifar100_modeldevries(run, do, **kwargs)
    config.exp.group_name = "supercifar_paper_sweep"
    config.data = cifar100_data_config(
        dataset="super_cifar100", img_size=DEFAULT_SMALL_IMG_SIZE
    )
    config.eval.query_studies = cifar100_query_config(
        dataset="super_cifar100", img_size=32
    )
    return config


def cnn_super_cifar100_modeldg(run: int, do: int, rew: float) -> Config:
    config = cnn_cifar100_modeldg(run, do, rew)
    config.exp.group_name = "supercifar_paper_sweep"
    config.data = cifar100_data_config(
        dataset="super_cifar100", img_size=DEFAULT_SMALL_IMG_SIZE
    )
    config.eval.query_studies = cifar100_query_config(
        dataset="super_cifar100", img_size=32
    )
    return config


def cnn_breeds(name: str) -> Config:
    config = cnn("breeds_paper_sweep", name=name)
    config.data = breeds_data_config()
    config.trainer.optimizer.init_args["init_args"]["lr"] = 0.1  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["weight_decay"] = 0.0001  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.fc_dim = 2048
    config.model.avg_pool = True
    config.eval.query_studies = breeds_query_config()
    return config


def cnn_breeds_modelconfidnet(run: int, do: int, **kwargs) -> Config:
    config = cnn_breeds(f"confidnet_bbresnet50_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 520
    config.trainer.num_epochs_backbone = 300
    config.trainer.learning_rate_confidnet = 0.0001
    config.trainer.learning_rate_confidnet_finetune = 1e-06
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 300  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.callbacks["training_stages"] = {}
    config.trainer.callbacks["training_stages"]["milestones"] = [300, 500]
    config.trainer.callbacks["training_stages"]["pretrained_backbone_path"] = None
    config.trainer.callbacks["training_stages"]["pretrained_confidnet_path"] = None
    config.trainer.callbacks["training_stages"]["disable_dropout_at_finetuning"] = True
    config.trainer.callbacks["training_stages"]["confidnet_lr_scheduler"] = False
    config.model.name = "confidnet_model"
    config.model.dropout_rate = do
    config.model.network.name = "confidnet_and_enc"
    config.model.network.backbone = "resnet50"
    config.eval.ext_confid_name = "tcp"
    return config


def cnn_breeds_modeldevries(run: int, do: int, **kwargs) -> Config:
    config = cnn_breeds(f"devries_bbresnet50_do{do}_run{run + 1}_rew2.2")
    config.trainer.num_epochs = 300
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 300  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = True  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dropout_rate = do
    config.model.dg_reward = -1
    config.model.network.name = "devries_and_enc"
    config.model.network.backbone = "resnet50"
    config.eval.ext_confid_name = "devries"
    return config


def cnn_breeds_modeldg(run: int, do: int, rew: float) -> Config:
    config = cnn_breeds(f"dg_bbresnet50_do{do}_run{run + 1}_rew{rew}")
    config.trainer.num_epochs = 350
    config.trainer.dg_pretrain_epochs = 50
    config.trainer.lr_scheduler.init_args["init_args"]["T_max"] = 350  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["nesterov"] = False  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.name = "devries_model"
    config.model.dg_reward = rew
    config.model.dropout_rate = do
    config.model.network.name = "resnet50"
    config.eval.ext_confid_name = "dg"
    return config


def vit(name: str) -> Config:
    config = Config(exp=ExperimentConfig(group_name="vit", name=name))
    config.trainer.num_epochs = None
    config.trainer.num_steps = 40000
    config.trainer.lr_scheduler_interval = "epoch"
    config.trainer.lr_scheduler = LRSchedulerConfig(
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
    )
    config.trainer.optimizer = OptimizerConfig(
        init_args={
            "class_path": "torch.optim.SGD",
            "init_args": {
                "dampening": 0.0,
                "momentum": 0.9,
                "nesterov": False,
                "maximize": False,
                "weight_decay": 0.0,
            },
        },
        class_path="fd_shifts.configs.OptimizerConfig",
    )
    config.trainer.batch_size = 128
    config.model.name = "vit_model"
    config.model.network.name = "vit"
    config.model.fc_dim = 512
    config.model.avg_pool = True
    config.eval.ext_confid_name = "maha"
    return config


def vit_modeldg(name: str) -> Config:
    config = vit(name)
    config.model.name = "devries_model"
    config.trainer.lr_scheduler_interval = "step"
    config.model.fc_dim = 768
    config.trainer.dg_pretrain_epochs = None
    config.eval.ext_confid_name = "dg"
    return config


def vit_wilds_animals_modeldg(run: int, lr: float, do: int, rew: float) -> Config:
    config = vit_modeldg(
        name=f"wilds_animals_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}",
    )
    config.data = wilds_animals_data_config(
        "wilds_animals", DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    config.trainer.num_steps = 60000
    config.trainer.batch_size = 512
    config.trainer.dg_pretrain_steps = 20000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 60000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.model.dg_reward = rew
    config.eval.query_studies = wilds_animals_query_config(
        DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    return config


def vit_wilds_camelyon_modeldg(run: int, lr: float, do: int, rew: float) -> Config:
    config = vit_modeldg(
        name=f"wilds_camelyon_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}",
    )
    config.data = wilds_camelyon_data_config(
        "wilds_camelyon", DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    config.trainer.num_steps = 60000
    config.trainer.dg_pretrain_steps = 20000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 60000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.model.dg_reward = rew
    config.eval.query_studies = wilds_camelyon_query_config(
        DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    return config


def vit_svhn_modeldg(run: int, lr: float, do: int, rew: float) -> Config:
    config = vit_modeldg(
        name=f"svhn_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}",
    )
    config.data = svhn_data_config("svhn", DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 60000
    config.trainer.dg_pretrain_steps = 20000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 60000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.model.dg_reward = rew
    config.eval.query_studies = svhn_query_config(
        "svhn", DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    return config


def vit_cifar10_modeldg(run: int, lr: float, do: int, rew: float) -> Config:
    config = vit_modeldg(
        name=f"cifar10_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}",
    )
    config.data = cifar10_data_config(img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 60000
    config.trainer.dg_pretrain_steps = 20000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 60000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.model.dg_reward = rew
    config.model.avg_pool = do == 0
    config.eval.query_studies = cifar10_query_config(DEFAULT_VIT_INFERENCE_IMG_SIZE)
    return config


def vit_cifar100_modeldg(run: int, lr: float, do: int, rew: float) -> Config:
    config = vit_modeldg(
        name=f"cifar100_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}",
    )
    config.data = cifar100_data_config(img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 15000
    config.trainer.batch_size = 512
    config.trainer.dg_pretrain_steps = 5000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 15000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.model.dg_reward = rew
    config.eval.query_studies = cifar100_query_config(DEFAULT_VIT_INFERENCE_IMG_SIZE)
    return config


def vit_super_cifar100_modeldg(run: int, lr: float, do: int, rew: float) -> Config:
    config = vit_cifar100_modeldg(run, lr, do, rew)
    config.exp.name = "super_" + config.exp.name
    config.data = cifar100_data_config(
        dataset="super_cifar100", img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    config.eval.query_studies = cifar100_query_config(
        dataset="super_cifar100", img_size=384
    )
    return config


def vit_breeds_modeldg(run: int, lr: float, do: int, rew: float) -> Config:
    config = vit_modeldg(
        name=f"breeds_modeldg_bbvit_lr{lr}_bs128_run{run}_do{do}_rew{rew}",
    )
    config.data = breeds_data_config("breeds", DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 60000
    config.trainer.dg_pretrain_steps = 20000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 60000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.model.dg_reward = rew
    config.eval.query_studies = breeds_query_config(DEFAULT_VIT_INFERENCE_IMG_SIZE)
    return config


def vit_wilds_animals_modelvit(run: int, lr: float, do: int, **kwargs) -> Config:
    config = vit(
        name=f"wilds_animals_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
    )
    config.data = wilds_animals_data_config(
        "wilds_animals", DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    config.trainer.num_steps = 40000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 40000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.eval.query_studies = wilds_animals_query_config(
        DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    return config


def vit_wilds_camelyon_modelvit(run: int, lr: float, do: int, **kwargs) -> Config:
    config = vit(
        name=f"wilds_camelyon_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
    )
    config.data = wilds_camelyon_data_config(
        "wilds_camelyon", DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    config.trainer.num_steps = 40000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 40000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.eval.query_studies = wilds_camelyon_query_config(
        DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    return config


def vit_svhn_modelvit(run: int, lr: float, do: int, **kwargs) -> Config:
    config = vit(
        name=f"svhn_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
    )
    config.data = svhn_data_config("svhn", DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 40000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 40000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.eval.query_studies = svhn_query_config(
        "svhn", DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    return config


def vit_cifar10_modelvit(run: int, lr: float, do: int, **kwargs) -> Config:
    config = vit(
        name=f"cifar10_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
    )
    config.data = cifar10_data_config(img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 40000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 40000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.model.avg_pool = do == 0
    config.eval.query_studies = cifar10_query_config(DEFAULT_VIT_INFERENCE_IMG_SIZE)
    return config


def vit_cifar100_modelvit(run: int, lr: float, do: int, **kwargs) -> Config:
    config = vit(
        name=f"cifar100_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
    )
    config.data = cifar100_data_config(img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 10000
    config.trainer.batch_size = 512
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 10000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.eval.query_studies = cifar100_query_config(DEFAULT_VIT_INFERENCE_IMG_SIZE)
    return config


def vit_super_cifar100_modelvit(run: int, lr: float, do: int, **kwargs) -> Config:
    config = vit_cifar100_modelvit(run, lr, do, **kwargs)
    config.exp.name = "super_" + config.exp.name
    config.data = cifar100_data_config(
        dataset="super_cifar100", img_size=DEFAULT_VIT_INFERENCE_IMG_SIZE
    )
    config.eval.query_studies = cifar100_query_config(
        dataset="super_cifar100", img_size=384
    )
    config.trainer.num_steps = 40000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 40000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    return config


def vit_breeds_modelvit(run: int, lr: float, do: int, **kwargs) -> Config:
    config = vit(
        name=f"breeds_modelvit_bbvit_lr{lr}_bs128_run{run}_do{do}_rew0",
    )
    config.data = breeds_data_config("breeds", DEFAULT_VIT_INFERENCE_IMG_SIZE)
    config.trainer.num_steps = 40000
    config.trainer.lr_scheduler.init_args["init_args"]["max_epochs"] = 40000  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.trainer.optimizer.init_args["init_args"]["lr"] = lr  # fmt: skip # pyright: ignore[reportOptionalMemberAccess] # noqa: E501
    config.model.dropout_rate = do
    config.eval.query_studies = breeds_query_config(DEFAULT_VIT_INFERENCE_IMG_SIZE)
    return config


def register(config_fn: Callable[..., Config], n_runs: int = 5, **kwargs) -> None:
    for run in range(n_runs):
        config = config_fn(**kwargs, run=run)
        __experiments[f"{config.exp.group_name}/{config.exp.name}"] = config


register(vit_svhn_modelvit, lr=0.03, do=1, rew=0)
register(vit_svhn_modelvit, lr=0.01, do=0, rew=0)
register(vit_svhn_modelvit, lr=0.01, do=1, rew=0)
register(vit_svhn_modeldg, lr=0.01, do=1, rew=2.2)
register(vit_svhn_modeldg, lr=0.01, do=1, rew=3)
register(vit_svhn_modeldg, lr=0.01, do=1, rew=6)
register(vit_svhn_modeldg, lr=0.01, do=1, rew=10)
register(vit_svhn_modeldg, lr=0.01, do=0, rew=2.2)
register(vit_svhn_modeldg, lr=0.01, do=0, rew=3)
register(vit_svhn_modeldg, lr=0.01, do=0, rew=6)
register(vit_svhn_modeldg, lr=0.01, do=0, rew=10)
register(vit_svhn_modeldg, lr=0.03, do=1, rew=2.2)
register(vit_svhn_modeldg, lr=0.03, do=1, rew=3)
register(vit_svhn_modeldg, lr=0.03, do=1, rew=6)
register(vit_svhn_modeldg, lr=0.03, do=1, rew=10)

register(vit_cifar10_modelvit, lr=3e-4, do=0, rew=0)
register(vit_cifar10_modelvit, lr=0.01, do=1, rew=0)
register(vit_cifar10_modeldg, lr=3e-4, do=0, rew=2.2)
register(vit_cifar10_modeldg, lr=0.01, do=1, rew=2.2)
register(vit_cifar10_modeldg, lr=3e-4, do=0, rew=3)
register(vit_cifar10_modeldg, lr=0.01, do=1, rew=3)
register(vit_cifar10_modeldg, lr=3e-4, do=0, rew=6)
register(vit_cifar10_modeldg, lr=0.01, do=1, rew=6)
register(vit_cifar10_modeldg, lr=3e-4, do=0, rew=10)
register(vit_cifar10_modeldg, lr=0.01, do=1, rew=10)

register(vit_cifar100_modelvit, lr=1e-2, do=0, rew=0)
register(vit_cifar100_modelvit, lr=1e-2, do=1, rew=0)
register(vit_cifar100_modeldg, lr=1e-2, do=0, rew=2.2)
register(vit_cifar100_modeldg, lr=1e-2, do=1, rew=2.2)
register(vit_cifar100_modeldg, lr=1e-2, do=0, rew=3)
register(vit_cifar100_modeldg, lr=1e-2, do=1, rew=3)
register(vit_cifar100_modeldg, lr=1e-2, do=0, rew=6)
register(vit_cifar100_modeldg, lr=1e-2, do=1, rew=6)
register(vit_cifar100_modeldg, lr=1e-2, do=0, rew=10)
register(vit_cifar100_modeldg, lr=1e-2, do=1, rew=10)
register(vit_cifar100_modeldg, lr=1e-2, do=0, rew=12)
register(vit_cifar100_modeldg, lr=1e-2, do=1, rew=12)
register(vit_cifar100_modeldg, lr=1e-2, do=0, rew=15)
register(vit_cifar100_modeldg, lr=1e-2, do=1, rew=15)
register(vit_cifar100_modeldg, lr=1e-2, do=0, rew=20)
register(vit_cifar100_modeldg, lr=1e-2, do=1, rew=20)

register(vit_super_cifar100_modelvit, lr=3e-3, do=0, rew=0)
register(vit_super_cifar100_modelvit, lr=1e-3, do=1, rew=0)
register(vit_super_cifar100_modeldg, lr=3e-3, do=0, rew=2.2)
register(vit_super_cifar100_modeldg, lr=1e-3, do=1, rew=2.2)
register(vit_super_cifar100_modeldg, lr=3e-3, do=0, rew=3)
register(vit_super_cifar100_modeldg, lr=1e-3, do=1, rew=3)
register(vit_super_cifar100_modeldg, lr=3e-3, do=0, rew=6)
register(vit_super_cifar100_modeldg, lr=1e-3, do=1, rew=6)
register(vit_super_cifar100_modeldg, lr=3e-3, do=0, rew=10)
register(vit_super_cifar100_modeldg, lr=1e-3, do=1, rew=10)
register(vit_super_cifar100_modeldg, lr=3e-3, do=0, rew=12)
register(vit_super_cifar100_modeldg, lr=1e-3, do=1, rew=12)
register(vit_super_cifar100_modeldg, lr=3e-3, do=0, rew=15)
register(vit_super_cifar100_modeldg, lr=1e-3, do=1, rew=15)
register(vit_super_cifar100_modeldg, lr=3e-3, do=0, rew=20)
register(vit_super_cifar100_modeldg, lr=1e-3, do=1, rew=20)

register(vit_wilds_animals_modelvit, lr=1e-3, do=0, rew=0)
register(vit_wilds_animals_modelvit, lr=1e-2, do=0, rew=0)
register(vit_wilds_animals_modelvit, lr=1e-2, do=1, rew=0)
register(vit_wilds_animals_modelvit, lr=3e-3, do=0, rew=0)
register(vit_wilds_animals_modelvit, lr=3e-3, do=1, rew=0)
register(vit_wilds_animals_modeldg, lr=1e-3, do=0, rew=2.2)
register(vit_wilds_animals_modeldg, lr=1e-3, do=0, rew=3)
register(vit_wilds_animals_modeldg, lr=1e-3, do=0, rew=6)
register(vit_wilds_animals_modeldg, lr=1e-3, do=0, rew=10)
register(vit_wilds_animals_modeldg, lr=1e-3, do=0, rew=15)
register(vit_wilds_animals_modeldg, lr=3e-3, do=0, rew=2.2)
register(vit_wilds_animals_modeldg, lr=3e-3, do=0, rew=3)
register(vit_wilds_animals_modeldg, lr=3e-3, do=0, rew=6)
register(vit_wilds_animals_modeldg, lr=3e-3, do=0, rew=10)
register(vit_wilds_animals_modeldg, lr=3e-3, do=0, rew=15)
register(vit_wilds_animals_modeldg, lr=3e-3, do=1, rew=2.2)
register(vit_wilds_animals_modeldg, lr=3e-3, do=1, rew=3)
register(vit_wilds_animals_modeldg, lr=3e-3, do=1, rew=6)
register(vit_wilds_animals_modeldg, lr=3e-3, do=1, rew=10)
register(vit_wilds_animals_modeldg, lr=3e-3, do=1, rew=15)

register(vit_wilds_camelyon_modelvit, lr=1e-3, do=0, rew=0)
register(vit_wilds_camelyon_modelvit, lr=3e-3, do=1, rew=0)
register(vit_wilds_camelyon_modeldg, lr=1e-3, do=0, rew=2.2)
register(vit_wilds_camelyon_modeldg, lr=1e-3, do=0, rew=3)
register(vit_wilds_camelyon_modeldg, lr=1e-3, do=0, rew=6)
register(vit_wilds_camelyon_modeldg, lr=1e-3, do=0, rew=10)
register(vit_wilds_camelyon_modeldg, lr=3e-3, do=1, rew=2.2)
register(vit_wilds_camelyon_modeldg, lr=3e-3, do=1, rew=3)
register(vit_wilds_camelyon_modeldg, lr=3e-3, do=1, rew=6)
register(vit_wilds_camelyon_modeldg, lr=3e-3, do=1, rew=10)

register(vit_breeds_modelvit, lr=3e-3, do=0, rew=0, n_runs=2)
register(vit_breeds_modelvit, lr=1e-3, do=0, rew=0, n_runs=2)
register(vit_breeds_modelvit, lr=1e-2, do=1, rew=0, n_runs=2)
register(vit_breeds_modeldg, lr=3e-3, do=0, rew=2.2, n_runs=2)
register(vit_breeds_modeldg, lr=3e-3, do=0, rew=3, n_runs=2)
register(vit_breeds_modeldg, lr=3e-3, do=0, rew=6, n_runs=2)
register(vit_breeds_modeldg, lr=3e-3, do=0, rew=10, n_runs=2)
register(vit_breeds_modeldg, lr=3e-3, do=0, rew=15, n_runs=2)
register(vit_breeds_modeldg, lr=1e-3, do=0, rew=2.2, n_runs=2)
register(vit_breeds_modeldg, lr=1e-3, do=0, rew=3, n_runs=2)
register(vit_breeds_modeldg, lr=1e-3, do=0, rew=6, n_runs=2)
register(vit_breeds_modeldg, lr=1e-3, do=0, rew=10, n_runs=2)
register(vit_breeds_modeldg, lr=1e-3, do=0, rew=15, n_runs=2)
register(vit_breeds_modeldg, lr=1e-2, do=1, rew=2.2, n_runs=2)
register(vit_breeds_modeldg, lr=1e-2, do=1, rew=3, n_runs=2)
register(vit_breeds_modeldg, lr=1e-2, do=1, rew=6, n_runs=2)
register(vit_breeds_modeldg, lr=1e-2, do=1, rew=10, n_runs=2)
register(vit_breeds_modeldg, lr=1e-2, do=1, rew=15, n_runs=2)

register(cnn_svhn_modeldevries, do=0)
register(cnn_svhn_modeldevries, do=1)
register(cnn_svhn_modelconfidnet, do=0)
register(cnn_svhn_modelconfidnet, do=1)
register(cnn_svhn_modeldg, do=0, rew=2.2)
register(cnn_svhn_modeldg, do=1, rew=2.2)
register(cnn_svhn_modeldg, do=0, rew=3)
register(cnn_svhn_modeldg, do=1, rew=3)
register(cnn_svhn_modeldg, do=0, rew=6)
register(cnn_svhn_modeldg, do=1, rew=6)
register(cnn_svhn_modeldg, do=0, rew=10)
register(cnn_svhn_modeldg, do=1, rew=10)

register(cnn_cifar10_modeldevries, do=0)
register(cnn_cifar10_modeldevries, do=1)
register(cnn_cifar10_modelconfidnet, do=0)
register(cnn_cifar10_modelconfidnet, do=1)
register(cnn_cifar10_modeldg, do=0, rew=2.2)
register(cnn_cifar10_modeldg, do=1, rew=2.2)
register(cnn_cifar10_modeldg, do=0, rew=3)
register(cnn_cifar10_modeldg, do=1, rew=3)
register(cnn_cifar10_modeldg, do=0, rew=6)
register(cnn_cifar10_modeldg, do=1, rew=6)
register(cnn_cifar10_modeldg, do=0, rew=10)
register(cnn_cifar10_modeldg, do=1, rew=10)

register(cnn_cifar100_modeldevries, do=0)
register(cnn_cifar100_modeldevries, do=1)
register(cnn_cifar100_modelconfidnet, do=0)
register(cnn_cifar100_modelconfidnet, do=1)
register(cnn_cifar100_modeldg, do=0, rew=2.2)
register(cnn_cifar100_modeldg, do=1, rew=2.2)
register(cnn_cifar100_modeldg, do=0, rew=3)
register(cnn_cifar100_modeldg, do=1, rew=3)
register(cnn_cifar100_modeldg, do=0, rew=6)
register(cnn_cifar100_modeldg, do=1, rew=6)
register(cnn_cifar100_modeldg, do=0, rew=10)
register(cnn_cifar100_modeldg, do=1, rew=10)
register(cnn_cifar100_modeldg, do=0, rew=12)
register(cnn_cifar100_modeldg, do=1, rew=12)
register(cnn_cifar100_modeldg, do=0, rew=15)
register(cnn_cifar100_modeldg, do=1, rew=15)
register(cnn_cifar100_modeldg, do=0, rew=20)
register(cnn_cifar100_modeldg, do=1, rew=20)

register(cnn_super_cifar100_modeldevries, do=0)
register(cnn_super_cifar100_modeldevries, do=1)
register(cnn_super_cifar100_modelconfidnet, do=0)
register(cnn_super_cifar100_modelconfidnet, do=1)
register(cnn_super_cifar100_modeldg, do=0, rew=2.2)
register(cnn_super_cifar100_modeldg, do=1, rew=2.2)
register(cnn_super_cifar100_modeldg, do=0, rew=3)
register(cnn_super_cifar100_modeldg, do=1, rew=3)
register(cnn_super_cifar100_modeldg, do=0, rew=6)
register(cnn_super_cifar100_modeldg, do=1, rew=6)
register(cnn_super_cifar100_modeldg, do=0, rew=10)
register(cnn_super_cifar100_modeldg, do=1, rew=10)
register(cnn_super_cifar100_modeldg, do=0, rew=12)
register(cnn_super_cifar100_modeldg, do=1, rew=12)
register(cnn_super_cifar100_modeldg, do=0, rew=15)
register(cnn_super_cifar100_modeldg, do=1, rew=15)
register(cnn_super_cifar100_modeldg, do=0, rew=20)
register(cnn_super_cifar100_modeldg, do=1, rew=20)

register(cnn_animals_modeldevries, do=0)
register(cnn_animals_modeldevries, do=1)
register(cnn_animals_modelconfidnet, do=0)
register(cnn_animals_modelconfidnet, do=1)
register(cnn_animals_modeldg, do=0, rew=2.2)
register(cnn_animals_modeldg, do=1, rew=2.2)
register(cnn_animals_modeldg, do=0, rew=3)
register(cnn_animals_modeldg, do=1, rew=3)
register(cnn_animals_modeldg, do=0, rew=6)
register(cnn_animals_modeldg, do=1, rew=6)
register(cnn_animals_modeldg, do=0, rew=10)
register(cnn_animals_modeldg, do=1, rew=10)
register(cnn_animals_modeldg, do=0, rew=15)
register(cnn_animals_modeldg, do=1, rew=15)

register(cnn_camelyon_modeldevries, do=0, n_runs=10)
register(cnn_camelyon_modeldevries, do=1, n_runs=10)
register(cnn_camelyon_modelconfidnet, do=0, n_runs=10)
register(cnn_camelyon_modelconfidnet, do=1, n_runs=10)
register(cnn_camelyon_modeldg, do=0, rew=2.2, n_runs=10)
register(cnn_camelyon_modeldg, do=1, rew=2.2, n_runs=10)
register(cnn_camelyon_modeldg, do=0, rew=3, n_runs=10)
register(cnn_camelyon_modeldg, do=1, rew=3, n_runs=10)
register(cnn_camelyon_modeldg, do=0, rew=6, n_runs=10)
register(cnn_camelyon_modeldg, do=1, rew=6, n_runs=10)
register(cnn_camelyon_modeldg, do=0, rew=10, n_runs=10)
register(cnn_camelyon_modeldg, do=1, rew=10, n_runs=10)

register(cnn_breeds_modeldevries, do=0, n_runs=2)
register(cnn_breeds_modeldevries, do=1, n_runs=2)
register(cnn_breeds_modelconfidnet, do=0, n_runs=2)
register(cnn_breeds_modelconfidnet, do=1, n_runs=2)
register(cnn_breeds_modeldg, do=0, rew=2.2, n_runs=2)
register(cnn_breeds_modeldg, do=1, rew=2.2, n_runs=2)
register(cnn_breeds_modeldg, do=0, rew=3, n_runs=2)
register(cnn_breeds_modeldg, do=1, rew=3, n_runs=2)
register(cnn_breeds_modeldg, do=0, rew=6, n_runs=2)
register(cnn_breeds_modeldg, do=1, rew=6, n_runs=2)
register(cnn_breeds_modeldg, do=0, rew=10, n_runs=2)
register(cnn_breeds_modeldg, do=1, rew=10, n_runs=2)
register(cnn_breeds_modeldg, do=0, rew=15, n_runs=2)
register(cnn_breeds_modeldg, do=1, rew=15, n_runs=2)


def get_experiment_config(name: str) -> Config:
    return __experiments[name]


def list_experiment_configs() -> list[str]:
    return sorted(__experiments.keys())
