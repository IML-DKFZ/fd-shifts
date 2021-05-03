from src.models.networks import svhn_small_conv
from src.models.networks import mnist_small_conv
from src.models.networks import mnist_mlp
from src.models.networks import vgg16
from src.models.networks import confidnet
from src.models.networks import devries_network
from src.models.networks import vgg13
from src.models.networks import zhang_network
from src.models.networks import resnet50_imagenet


def get_network(network_name):
    """
        Return a new instance of a backbone
    """

    # Available models
    network_factory = {
        "svhn_small_conv": svhn_small_conv.SmallConv, # todo make explciit arguments!!
        "mnist_small_conv": mnist_small_conv.SmallConv, # todo make explciit arguments!!
        "mnist_mlp": mnist_mlp.MLP, # todo make explciit arguments!!
        # "vgg16": vgg16.VGG16, # todo make explciit arguments!!
        "confidnet_and_enc": confidnet.ConfidNetAndEncoder, # todo make explciit arguments!!
        "devries_and_enc": devries_network.DeVriesAndEncoder, # todo make explciit arguments!!
        "vgg13": vgg13.VGG13, # todo make explciit arguments!!
        "vgg16": vgg13.VGG13, # todo make explciit arguments!!
        "zhang_and_enc": zhang_network.ZhangAndEncoder, # todo make explciit arguments!!
        "resnet50": resnet50_imagenet.resnet50, # todo make explciit arguments!!
    }

    return network_factory[network_name]
