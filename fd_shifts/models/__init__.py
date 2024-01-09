import pytorch_lightning as pl

from fd_shifts.models import clip_model, confidnet_model, devries_model, vit_model

_model_factory: dict[str, type[pl.LightningModule]] = {
    "confidnet_model": confidnet_model.Module,
    "devries_model": devries_model.net,
    "vit_model": vit_model.net,
    "clip_model": clip_model.ClipOodModel,
}


def register_model(model_name: str, model: type[pl.LightningModule]) -> None:
    """Register a new model class

    Args:
        model_name (str):
        model (type[pl.LightningModule]):
    """
    _model_factory[model_name] = model


def get_model(model_name: str) -> type[pl.LightningModule]:
    """
    Args:
        model_name (str): name as string

    Returns:
        a new instance of model
    """
    return _model_factory[model_name]


def model_exists(model_name: str) -> bool:
    """
    Args:
        model_name (str): name as string

    Returns:
        a new instance of model
    """
    return model_name in _model_factory
