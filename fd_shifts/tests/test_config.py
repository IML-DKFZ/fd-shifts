import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional, cast

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf
from rich import print as pprint

from fd_shifts import configs


@pytest.fixture
def mock_env_if_missing(monkeypatch) -> None:
    monkeypatch.setenv(
        "EXPERIMENT_ROOT_DIR", os.getenv("EXPERIMENT_ROOT_DIR", default="./experiments")
    )
    monkeypatch.setenv(
        "DATASET_ROOT_DIR", os.getenv("DATASET_ROOT_DIR", default="./data")
    )


def initialize_hydra(overrides: list[str]) -> DictConfig:
    """Takes the place of hydra.main"""
    default_overrides = [
        "hydra.job.num=0",
        "hydra.job.id=0",
        "hydra.runtime.output_dir=.",
        "hydra.hydra_help.hydra_help=''",
    ]

    with initialize_config_module(version_base=None, config_module="fd_shifts.configs"):
        cfg = compose(
            "config", overrides=overrides + default_overrides, return_hydra_config=True
        )
        HydraConfig.instance().set_config(cfg)

    return cfg


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ([], nullcontext()),
        (["exp.log_path=test"], nullcontext()),
        (["exp.global_seed=1234"], nullcontext()),
        (["pkgversion=0.0.1+aefe5c8"], pytest.raises(ValueError)),
        (["trainer.val_split=foo"], pytest.raises(ConfigCompositionException)),
        (["eval.query_studies.iid_study=svhn"], nullcontext()),
        (
            ["eval.query_studies.iid_study=doesnt_exist_dataset"],
            pytest.raises(ValueError),
        ),
    ],
)
def test_validation(
    overrides: list[str], expected: AbstractContextManager, mock_env_if_missing: Any
):
    with expected:
        configs.init()

        cfg = initialize_hydra(overrides)

        print(type(cfg))
        cfg = OmegaConf.to_object(cfg)
        print(cfg.pkgversion)
        print(cfg.exp.global_seed)
        pprint(OmegaConf.to_yaml(cfg, resolve=True))
        print(type(cfg))
        assert cfg.exp.root_dir == Path(os.getenv("EXPERIMENT_ROOT_DIR"))


@pytest.mark.parametrize(
    ("study",),
    [
        ("breeds_vit_study",),
        ("cifar100_vit_study",),
        ("cifar10_vit_study",),
        ("confidnet",),
        ("deepgamblers",),
        ("devries",),
        ("super_cifar100_vit_study",),
        ("svhn_openset_vit_study",),
        ("svhn_vit_study",),
        ("wilds_animals_openset_vit_study",),
        ("wilds_animals_vit_study",),
        ("wilds_camelyon_vit_study",),
    ],
)
def test_existing_studies(study: str, mock_env_if_missing: Any):
    configs.init()
    overrides = [f"study={study}"]

    cfg = initialize_hydra(overrides)

    print(type(cfg))
    cfg = OmegaConf.to_object(cfg)
    pprint(OmegaConf.to_yaml(cfg, resolve=False))
    print(type(cfg))


def _normalize_dataset_name(dataset: str):
    return dataset.replace("_data", "").replace("_384", "").replace("_ood_test", "")


@pytest.mark.parametrize(
    ("dataset",),
    [
        ("breeds_384_data",),
        ("breeds_data",),
        ("breeds_ood_test_384_data",),
        ("breeds_ood_test_data",),
        ("cifar100_384_data",),
        ("cifar100_data",),
        ("cifar10_384_data",),
        ("cifar10_data",),
        ("corrupt_cifar100_384_data",),
        ("corrupt_cifar100_data",),
        ("corrupt_cifar10_384_data",),
        ("corrupt_cifar10_data",),
        ("super_cifar100_384_data",),
        ("super_cifar100_data",),
        ("svhn_384_data",),
        ("svhn_data",),
        ("svhn_openset_384_data",),
        ("svhn_openset_data",),
        ("tinyimagenet_384_data",),
        ("tinyimagenet_resize_data",),
        ("wilds_animals_384_data",),
        ("wilds_animals_data",),
        ("wilds_animals_ood_test_384_data",),
        ("wilds_animals_ood_test_data",),
        ("wilds_animals_openset_384_data",),
        ("wilds_animals_openset_data",),
        ("wilds_camelyon_384_data",),
        ("wilds_camelyon_data",),
        ("wilds_camelyon_ood_test_384_data",),
        ("wilds_camelyon_ood_test_data",),
    ],
)
def test_existing_datasets(dataset: str, mock_env_if_missing: Any):
    configs.init()
    overrides = [f"data={dataset}"]

    dcfg = initialize_hydra(overrides)

    print(type(dcfg))
    cfg: configs.Config = cast(
        configs.Config, OmegaConf.to_object(dcfg)
    )  # only affects the linter
    pprint(OmegaConf.to_yaml(cfg, resolve=False))
    assert isinstance(cfg, configs.Config)  # runtime type check
    assert cfg.data.dataset == _normalize_dataset_name(dataset)
