from models import small_conv


def get_model(cf):
    """
        Return a new instance of model
    """

    # Available models
    model_factory = {
        "small_conv": small_conv.net,
    }

    return model_factory[cf.model.name](cf)
