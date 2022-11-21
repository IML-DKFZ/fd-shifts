from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, TypeAlias, TypeVar, TYPE_CHECKING

from torch import nn

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
    # Available models
    from fd_shifts.models.networks import (
        confidnet,
        devries_network,
        dgvgg,
        mnist_mlp,
        mnist_small_conv,
        resnet50_imagenet,
        svhn_small_conv,
        vgg,
        vgg16,
        vgg_devries,
        vit,
        zhang_network,
    )

    return {
        "svhn_small_conv": svhn_small_conv.SmallConv,
        # "mnist_small_conv": mnist_small_conv.SmallConv,
        # "mnist_mlp": mnist_mlp.MLP,
        "confidnet_and_enc": confidnet.ConfidNetAndEncoder,
        "devries_and_enc": devries_network.DeVriesAndEncoder,
        "vgg13": vgg.VGG,
        "vgg16": vgg.VGG,
        # "vgg_old": vgg16.VGG16,
        # "vgg_devries": vgg_devries.VGG13,
        # "zhang_and_enc": zhang_network.ZhangAndEncoder,
        # "zhang_backbone": zhang_network.ZhangBackbone,
        "resnet50": resnet50_imagenet.resnet50,
        # "dgvgg": dgvgg.VGG,
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
