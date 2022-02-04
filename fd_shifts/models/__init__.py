from fd_shifts.models import det_mcd_model
from fd_shifts.models import confidnet_model
from fd_shifts.models import zhang_model
from fd_shifts.models import devries_model
from fd_shifts.models import vit_model


def get_model(model_name):
    """
    Return a new instance of model
    """

    # Available models
    model_factory = {
        "det_mcd_model": det_mcd_model.net,
        "confidnet_model": confidnet_model.net,
        "zhang_model": zhang_model.net,
        "devries_model": devries_model.net,
        "vit_model": vit_model.net,
    }

    return model_factory[model_name]