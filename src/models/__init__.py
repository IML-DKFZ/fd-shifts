from src.models import det_mcd_model
from src.models import confidnet_model
from src.models import zhang_model
from src.models import devries_model


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
    }

    return model_factory[model_name]
