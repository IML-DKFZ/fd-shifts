import os
from collections.abc import Iterable

import pydantic
import pytest
import pytorch_lightning as pl
from rich import print

from fd_shifts import configs, models


@pytest.fixture
def mock_env_if_missing(monkeypatch) -> None:
    monkeypatch.setenv(
        "EXPERIMENT_ROOT_DIR", os.getenv("EXPERIMENT_ROOT_DIR", default="./experiments")
    )
    monkeypatch.setenv(
        "DATASET_ROOT_DIR", os.getenv("DATASET_ROOT_DIR", default="./data")
    )


class MyModel(pl.LightningModule):
    pass


def test_register_model(mock_env_if_missing):
    configs.init()

    config = configs.Config.with_defaults()

    models.register_model("my_model", MyModel)

    config.model.name = "my_model"

    print(config)
