from fd_shifts.models import (
    confidnet_model,
    confidnet_model_mod,
    cross_entropy,
    det_mcd_model,
    devries_model,
    devries_model_mod,
    isic_model,
    vit_model,
    zhang_model,
)


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
        "isic_model": isic_model.net,
        "devries_model_mod": devries_model_mod.net,
        "confidnet_model_mod": confidnet_model_mod.net,
        "cross_entropy": cross_entropy.net,
    }

    return model_factory[model_name]
