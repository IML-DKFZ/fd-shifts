from src.models.backbones import small_conv


def get_backbone(backbone_name):
    """
        Return a new instance of a backbone
    """

    # Available models
    backbone_factory = {
        "small_conv": small_conv.Encoder, # todo make explciit arguments!!
    }

    return backbone_factory[backbone_name]
