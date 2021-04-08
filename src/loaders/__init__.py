import src.loaders.svhn_loader
import src.loaders.mnist_loader
import src.loaders.cifar10_loader
import src.loaders.cifar100_loader

def get_loader(cf):
    """
        Return a new instance of dataset loader
    """

    # Available models
    data_loader_factory = {
        "svhn": svhn_loader.DataLoader,
        "mnist": mnist_loader.DataLoader,
        "cifar10": cifar10_loader.DataLoader,
        "cifar100": cifar100_loader.DataLoader,
    }

    return data_loader_factory[cf.data.dataset](cf)
