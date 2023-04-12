from fd_shifts.models.networks import (
    confidnet,
    densenet121,
    densenet161,
    devries_network,
    dgvgg,
    efficientnetb4,
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


def get_network(network_name):
    """
    Return a new instance of a backbone
    """

    # Available models
    network_factory = {
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
        "efficientnetb4": efficientnetb4.EfficientNetb4,
        "densenet121": densenet121.Densenet121,
        "densenet161": densenet161.Densenet161,
    }

    return network_factory[network_name]
