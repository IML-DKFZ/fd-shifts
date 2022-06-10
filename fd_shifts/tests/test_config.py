from pathlib import Path
from typing import Optional

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich import print as pprint

from fd_shifts import configs

# This would normally get set by hydra.main
default_overrides = [
    "hydra.job.num=0",
    "hydra.job.id=0",
    "hydra.runtime.output_dir=.",
    "hydra.hydra_help.hydra_help=''",
]


def initialize_hydra(overrides: list[str]) -> DictConfig:
    """Takes the place of hydra.main"""
    with initialize_config_module(version_base=None, config_module="fd_shifts.configs"):
        cfg = compose(
            "config", overrides=overrides + default_overrides, return_hydra_config=True
        )
        HydraConfig.instance().set_config(cfg)

    return cfg


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["exp.log_path=test"],
        pytest.param(
            ["trainer.val_split=foo"],
            marks=pytest.mark.xfail(reason="foo is not a valid val_split"),
        ),
    ],
)
def test_schema_with_old(overrides: list[str]):
    configs.init()

    cfg = initialize_hydra(overrides)

    print(type(cfg))
    cfg = OmegaConf.to_object(cfg)
    pprint(OmegaConf.to_yaml(cfg, resolve=False))
    print(type(cfg))


# @hydra.main(config_path="../configs", config_name="config")
# def main(cfg):
#
#     pprint(OmegaConf.to_yaml(cfg))
#     OmegaConf.resolve(cfg)
#     pprint(OmegaConf.to_yaml(cfg))
#     OmegaConf.to_object(cfg)
#     print("DONE")
#
#
# if __name__ == "__main__":
#     configs.init()
#     main()
