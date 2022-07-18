import os
from pathlib import Path
from typing import Any, cast

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich import print as pprint

from fd_shifts import configs, models
from fd_shifts.loaders.abstract_loader import AbstractDataLoader
from fd_shifts.models.callbacks import get_callbacks
from fd_shifts.utils import exp_utils


@pytest.fixture
def mock_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXPERIMENT_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("DATASET_ROOT_DIR", str(Path("~/Data").expanduser()))


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


@pytest.mark.memory_heavy
@pytest.mark.parametrize(
    ("study",),
    [
        # ("breeds_vit_study",),
        # ("cifar100_vit_study",),
        ("cifar10_vit_study",),
        ("confidnet",),
        ("deepgamblers",),
        ("devries",),
        # ("super_cifar100_vit_study",),
        # ("svhn_openset_vit_study",),
        # ("svhn_vit_study",),
        # ("wilds_animals_openset_vit_study",),
        # ("wilds_animals_vit_study",),
        # ("wilds_camelyon_vit_study",),
    ],
)
def test_model_creation(study: str, snapshot: Any, mock_env_if_missing: Any):
    exp_utils.set_seed(1234)

    configs.init()
    overrides = [f"study={study}"]
    dcfg = initialize_hydra(overrides)

    print(type(dcfg))
    cfg: configs.Config = cast(
        configs.Config, OmegaConf.to_object(dcfg)
    )  # only affects the linter

    model = models.get_model(cfg.model.name)(cfg)
    test_image = torch.ones(16, *cfg.data.img_size[::-1])
    assert model(test_image) == snapshot


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.memory_heavy
@pytest.mark.parametrize(
    ("study",),
    [
        # ("breeds_vit_study",),
        # ("cifar100_vit_study",),
        ("cifar10_vit_study",),
        ("confidnet",),
        ("deepgamblers",),
        ("devries",),
        # ("super_cifar100_vit_study",),
        # ("svhn_openset_vit_study",),
        # ("svhn_vit_study",),
        # ("wilds_animals_openset_vit_study",),
        # ("wilds_animals_vit_study",),
        # ("wilds_camelyon_vit_study",),
    ],
)
def test_model_training(study: str, snapshot: Any, tmp_path: Path, mock_env: None):
    assert str(tmp_path) == os.environ["EXPERIMENT_ROOT_DIR"]

    exp_utils.set_seed(1234)

    configs.init()
    overrides = [f"study={study}"]
    dcfg = initialize_hydra(overrides)

    print(type(dcfg))
    cfg: configs.Config = cast(
        configs.Config, OmegaConf.to_object(dcfg)
    )  # only affects the linter

    cfg.trainer.batch_size = 4
    cfg.data.num_workers = 0
    cfg.exp.group_dir.mkdir()
    cfg.exp.dir.mkdir()
    cfg.exp.version_dir.mkdir()
    cfg.test.dir.mkdir()

    def _filter_unstable_line(line: str) -> bool:
        return not (
            "- pytest-" in line  # tmppaths
            or "pkgversion" in line  # commit being tested
        )

    assert "\n".join(filter(_filter_unstable_line, OmegaConf.to_yaml(cfg).split("\n"))) == snapshot(name="config")

    datamodule = AbstractDataLoader(cfg)
    model = models.get_model(cfg.model.name)(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.num_epochs,
        max_steps=cfg.trainer.num_steps,
        fast_dev_run=5,
        callbacks=get_callbacks(cfg),
        deterministic=True,
        gradient_clip_val=1,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    test_results = list(map(lambda p: p.relative_to(tmp_path), tmp_path.rglob("*")))
    assert test_results == snapshot(name="test_results_file_list")
    for file in filter(lambda p: p.is_file(), test_results):
        assert file.stat().st_size > 0

    test_image = torch.ones(16, *cfg.data.img_size[::-1])
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = model(test_image)

    if isinstance(output, tuple):
        output = output[0]

    assert not torch.isnan(output).any()
    assert not (output == 0).all()
    assert output == snapshot


# TODO: Test checkpointing and loading
# TODO: Check output files
