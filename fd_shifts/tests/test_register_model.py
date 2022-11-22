from collections.abc import Iterable
import pydantic
from rich import print
from fd_shifts import models, configs
import pytorch_lightning as pl

def _enable_validation(conf):
    if hasattr(conf, "__pydantic_run_validation__"):
        conf.__pydantic_run_validation__ = True

    if not isinstance(conf, Iterable):
        return

    for _, v in conf:
        if hasattr(v, "__pydantic_run_validation__"):
            _enable_validation(v)


class MyModel(pl.LightningModule):
    pass

def test_register_model():
    configs.init()

    config = configs.Config.with_defaults()

    models.register_model("my_model", MyModel)

    config.model.name = "my_model"

    print(config)
