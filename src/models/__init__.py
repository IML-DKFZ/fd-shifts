from src.models import default_classifier


def get_model(model_name):
    """
        Return a new instance of model
    """

    # Available models
    model_factory = {
        "default_classifier": default_classifier.net,
    }

    return model_factory[model_name]
