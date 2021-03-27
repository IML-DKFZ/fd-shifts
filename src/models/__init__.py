from src.models import small_conv


def get_model(model_name):
    """
        Return a new instance of model
    """

    # Available models
    model_factory = {
        "small_conv": small_conv.net,
    }

    return model_factory[model_name]
