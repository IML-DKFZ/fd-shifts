import os
from collections.abc import Iterable

import pydantic
import pytest
import pytorch_lightning as pl
from rich import print

from fd_shifts import configs, models
from fd_shifts.tests.utils import mock_env_if_missing


class MyModel(pl.LightningModule):
    pass


@pytest.mark.skip("TODO: does nothing, remove or improve")
def test_register_model(mock_env_if_missing):
    configs.init()

    config = configs.Config.with_defaults()

    models.register_model("my_model", MyModel)

    config.model.name = "my_model"

    print(config)
