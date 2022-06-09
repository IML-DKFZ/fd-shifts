from pathlib import Path
from typing import Optional

import hydra
import pytest
from hydra import compose, initialize, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from rich import print as pprint

import fd_shifts.configs as configs


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ([], None),
        # pytest.param(["exp.log_path=test"], None, marks=pytest.mark.xfail),
        (["exp.log_path=test"], None),
    ],
)
def test_schema_with_old(overrides: list[str], expected: Optional[int]):

    store = ConfigStore.instance()
    store.store(name="config_schema", node=configs.Config)
    store.store(group="data", name="data_schema", node=configs.DataConfig)

    with initialize_config_module(version_base=None, config_module="fd_shifts.configs"):
        cfg = compose("config", overrides=overrides)
        pprint(OmegaConf.to_yaml(cfg))

    # assert False


# TODO: Write tests with weird configs
