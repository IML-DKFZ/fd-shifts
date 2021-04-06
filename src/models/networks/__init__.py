from src.models.networks import small_conv
from src.models.networks import small_conv_debug
from src.models.networks import confidnet_small_conv


def get_network(network_name):
    """
        Return a new instance of a backbone
    """

    # Available models
    network_factory = {
        "small_conv": small_conv.SmallConv, # todo make explciit arguments!!
        "small_conv_debug": small_conv_debug.SmallConv, # todo make explciit arguments!!
        "confidnet_small_conv_and_enc": confidnet_small_conv.ConfidNetAndENcoder, # todo make explciit arguments!!
    }

    return network_factory[network_name]
