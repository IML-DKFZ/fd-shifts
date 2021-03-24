import imlone.loaders.svhn_loader

def get_loader(cfg):
    """
        Return a new instance of dataset loader
    """

    # Available models
    data_loader_factory = {
        "svhn": svhn_loader.DataLoader,
    }

    return data_loader_factory[cfg.data.dataset](cfg)
