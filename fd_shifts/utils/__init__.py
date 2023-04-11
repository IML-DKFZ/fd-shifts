import importlib


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
