from imlone.models import small_conv_pl


def get_model(cf):
    """
        Return a new instance of model
    """

    # Available models
    model_factory = {
        "small_conv_pl": small_conv_pl.net,
    }

    return model_factory[cf.model.name](cf)