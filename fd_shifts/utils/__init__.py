import importlib
import json
from dataclasses import asdict

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic.json import pydantic_encoder


def __to_dict(obj):
    if isinstance(obj, DictConfig) or isinstance(obj, ListConfig):
        return OmegaConf.to_container(obj)
    return pydantic_encoder(obj)


def to_dict(obj):
    # s = json.dumps(obj, default=__to_dict)
    # return json.loads(s)
    return asdict(obj)


def instantiate_from_str(name, *args, **kwargs):
    """"""
    module, class_name = name.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module, package=None), class_name)(
            *args, **kwargs
        )
    except Exception as err:
        raise ValueError(
            f"Failed to instantiate '{name}'. It may need to be registered via a "
            "suitable 'register_*' function."
        ) from err
