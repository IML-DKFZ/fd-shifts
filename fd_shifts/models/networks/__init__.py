from __future__ import annotations

from typing import Callable, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from fd_shifts import configs
    from fd_shifts.models.networks import network

    NetworkFactoryType: TypeAlias = Callable[
        [
            configs.Config,
        ],
        network.Network,
    ]


def _get_network_factory() -> dict[str, NetworkFactoryType]:
    from fd_shifts.models.networks import (
        confidnet,
        devries_network,
        resnet50_imagenet,
        svhn_small_conv,
        vgg,
        vit,
    )

    return {
        "svhn_small_conv": svhn_small_conv.SmallConv,
        "confidnet_and_enc": confidnet.ConfidNetAndEncoder,
        "devries_and_enc": devries_network.DeVriesAndEncoder,
        "vgg13": vgg.VGG,
        "vgg16": vgg.VGG,
        "resnet50": resnet50_imagenet.resnet50,
        "vit": vit.ViT,
    }


def get_network(network_name: str) -> NetworkFactoryType:
    """
    Args:
        network_name (str): name of the network

    Returns:
        a new instance of a backbone
    """
    return _get_network_factory()[network_name]


def network_exists(network_name: str) -> bool:
    """
    Args:
        network_name (str): name of the network

    Returns:
        whether the network exists or not
    """
    return network_name in _get_network_factory()
