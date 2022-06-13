from fd_shifts.models.networks import svhn_small_conv
from fd_shifts.models.networks import mnist_small_conv
from fd_shifts.models.networks import mnist_mlp
from fd_shifts.models.networks import vgg16
from fd_shifts.models.networks import confidnet
from fd_shifts.models.networks import devries_network
from fd_shifts.models.networks import vgg
from fd_shifts.models.networks import zhang_network
from fd_shifts.models.networks import resnet50_imagenet
from fd_shifts.models.networks import vgg_devries
from fd_shifts.models.networks import dgvgg
from fd_shifts.models.networks import vit

# TODO: Error handling
# TODO: Make explicit arguments
# TODO: Throw out unused networks
# TODO: Make easily extendable -> registry


# Available models
_network_factory = {
    "svhn_small_conv": svhn_small_conv.SmallConv,  # todo make explciit arguments!!
    "mnist_small_conv": mnist_small_conv.SmallConv,  # todo make explciit arguments!!
    "mnist_mlp": mnist_mlp.MLP,  # todo make explciit arguments!!
    # "vgg16": vgg16.VGG16, # todo make explciit arguments!!
    "confidnet_and_enc": confidnet.ConfidNetAndEncoder,  # todo make explciit arguments!!
    "devries_and_enc": devries_network.DeVriesAndEncoder,  # todo make explciit arguments!!
    "vgg13": vgg.VGG,  # todo make explciit arguments!!
    "vgg16": vgg.VGG,  # todo make explciit arguments!!
    "vgg_old": vgg16.VGG16,  # todo make explciit arguments!!
    "vgg_devries": vgg_devries.VGG13,  # todo make explciit arguments!!
    "zhang_and_enc": zhang_network.ZhangAndEncoder,  # todo make explciit arguments!!
    "zhang_backbone": zhang_network.ZhangBackbone,  # todo make explciit arguments!!
    "resnet50": resnet50_imagenet.resnet50,  # todo make explciit arguments!!
    "dgvgg": dgvgg.VGG,  # todo make explciit arguments!!
    "vit": vit.ViT,
}

def get_network(network_name):
    """
    Return a new instance of a backbone
    """
    return _network_factory[network_name]

def network_exists(network_name):
    """
    Return a new instance of a backbone
    """
    return network_name in _network_factory
