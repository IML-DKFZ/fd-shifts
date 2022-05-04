import io
import os
import pickle
from typing import Any, Callable, Optional, Tuple, TypeVar

import numpy as np
import torchvision
from medmnist.info import DEFAULT_ROOT, HOMEPAGE, INFO
from PIL import Image
from robustness.tools.breeds_helpers import (
    ClassHierarchy,
    make_entity13,
    print_dataset_info,
)
from robustness.tools.folder import ImageFolder
from robustness.tools.helpers import get_label_mapping
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.wilds_dataset import WILDSSubset

from fd_shifts import logger
from fd_shifts.analysis import eval_utils
from fd_shifts.loaders import breeds_hierarchies

# def dataset_exists(name: str) -> bool:
#     """Check if dataset with name is registered
#
#     Args:
#         name (str): name of the dataset
#
#     Returns:
#         True if it exists
#     """
#     return name in _dataset_factory
#
#
# def register_dataset(name: str, dataset: type) -> None:
#     """Register a new dataset
#
#     Args:
#         name (str): name to register under
#         dataset (type): dataset class to register
#     """
#     _dataset_factory[name] = dataset


def get_dataset(
    name: str,
    root: str,
    train: bool,
    download: bool,
    transform: Callable,
    kwargs: dict[str, Any],
) -> Any:
    """Return a new instance of a dataset

    Args:
        name (str): name of the dataset
        root (str): where it is stored on disk
        train (bool): whether to load the train split
        download (bool): whether to attempt to download if it is not in root
        transform (Callable): transforms to apply to loaded data
        kwargs (dict[str, Any]): other kwargs to pass on

    Returns:
        dataset instance
    """
    _dataset_factory: dict[str, type] = {
        "svhn": datasets.SVHN,
        "svhn_384": datasets.SVHN,
        "svhn_openset": SVHNOpenSet,
        "svhn_openset_384": SVHNOpenSet,
        "tinyimagenet_384": datasets.ImageFolder,
        "tinyimagenet_resize": datasets.ImageFolder,
        "emnist_byclass": datasets.EMNIST,
        "emnist_bymerge": datasets.EMNIST,
        "emnist_balanced": datasets.EMNIST,
        "emnist_letters": datasets.EMNIST,
        "emnist_digits": datasets.EMNIST,
        "emnist_mnist": datasets.EMNIST,
        "med_mnist_path": PathMNIST,
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "cifar10_384": datasets.CIFAR10,
        "cifar100_384": datasets.CIFAR100,
        "super_cifar100": SuperCIFAR100,
        "super_cifar100_384": SuperCIFAR100,
        "corrupt_cifar100": CorruptCIFAR,
        "corrupt_cifar100_384": CorruptCIFAR,
        "corrupt_cifar10": CorruptCIFAR,
        "corrupt_cifar10_384": CorruptCIFAR,
        "breeds": BREEDImageNet,
        "breeds_ood_test": BREEDImageNet,
        "breeds_384": BREEDImageNet,
        "breeds_ood_test_384": BREEDImageNet,
        "wilds_animals": WILDSAnimals,
        "wilds_animals_ood_test": WILDSAnimals,
        "wilds_animals_384": WILDSAnimals,
        "wilds_animals_ood_test_384": WILDSAnimals,
        "wilds_animals_openset": WILDSAnimalsOpenSet,
        "wilds_animals_openset_384": WILDSAnimalsOpenSet,
        "wilds_camelyon": WILDSCamelyon,
        "wilds_camelyon_384": WILDSCamelyon,
        "wilds_camelyon_ood_test": WILDSCamelyon,
        "wilds_camelyon_ood_test_384": WILDSCamelyon,
    }
    pass_kwargs = {
        "root": root,
        "train": train,
        "download": download,
        "transform": transform,
    }
    if name.startswith("svhn"):
        pass_kwargs = {
            "root": root,
            "split": "train" if train else "test",
            "download": download,
            "transform": transform,
        }
    if name.startswith("med_mnist"):
        pass_kwargs = {
            "root": root,  # find a way to set this flexible!
            "split": "train" if train else "test",
            "download": download,
            "transform": transform,
        }
    if "openset" in name:
        pass_kwargs["out_classes"] = kwargs["out_classes"]
    if name == "tinyimagenet" or name == "tinyimagenet_384":
        pass_kwargs = {"root": os.path.join(root, "test"), "transform": transform}
    if name == "tinyimagenet_resize":
        pass_kwargs = {"root": root, "transform": transform}

    elif "breeds" in name:
        if name == "breeds":
            split = "train" if train else "id_test"
        elif name == "breeds_ood_test":
            split = "ood_test"
        elif name == "breeds_384":
            split = "train" if train else "id_test"
        elif name == "breeds_ood_test_384":
            split = "ood_test"
        logger.debug("CHECK SPLIT {} {}", name, split)
        pass_kwargs = {
            "root": root,
            "split": split,
            "download": download,
            "transform": transform,
            "kwargs": kwargs,
        }

    if "wilds" in name:
        if name == "wilds_animals":
            split = "train" if train else "id_test"
        elif name == "wilds_animals_ood_test":
            split = "test"
        elif name == "wilds_animals_384":
            split = "train" if train else "id_test"
        elif name == "wilds_animals_ood_test_384":
            split = "test"
        elif name == "wilds_animals_openset":
            split = "train" if train else "id_test"
        elif name == "wilds_animals_openset_384":
            split = "train" if train else "id_test"
        elif name == "wilds_camelyon":
            split = "train" if train else "id_val"  # currently for chamelyon
        elif name == "wilds_camelyon_ood_test":
            split = "test"
        elif name == "wilds_camelyon_384":
            split = "train" if train else "id_val"  # currently for chamelyon
        elif name == "wilds_camelyon_ood_test_384":
            split = "test"
        return _dataset_factory[name](**pass_kwargs).get_subset(
            split, frac=1.0, transform=transform
        )
    if "emnist" in name:
        if name == "emnist_byclass":
            split = "byclass"
        elif name == "emnist_bymerge":
            split = "bymerge"
        elif name == "emnist_balanced":
            split = "balanced"
        elif name == "emnist_letters":
            split = "letters"
        elif name == "emnist_digits":
            split = "digits"
        elif name == "emnist_mnist":
            split = "mnist"
        dataset = _dataset_factory[name](split=split, **pass_kwargs)
        return dataset
    else:
        return _dataset_factory[name](**pass_kwargs)


class MedMNIST_mod(Dataset):
    flag = ...

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        as_rgb=False,
        root=DEFAULT_ROOT,
    ):
        """dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        """

        self.info = INFO[self.flag]
        root = os.path.expanduser(root)  # recognize ~ as home directory
        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        if download:
            self.download()

        if not os.path.exists(os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError(
                "Dataset not found. " + " You can set `download=True` to download it"
            )

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        self.data, self.targets = self._load_data()

        if self.split == "train":
            self.imgs = npz_file["train_images"]
            self.labels = npz_file["train_labels"]
        elif self.split == "val":
            self.imgs = npz_file["val_images"]
            self.labels = npz_file["val_labels"]
        elif self.split == "test":
            self.imgs = npz_file["test_images"]
            self.labels = npz_file["test_labels"]
        else:
            raise ValueError

    def _load_data(self):
        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))
        if self.split == "train":
            self.imgs = npz_file["train_images"]
            self.labels = npz_file["train_labels"]
        elif self.split == "val":
            self.imgs = npz_file["val_images"]
            self.labels = npz_file["val_labels"]
        elif self.split == "test":
            self.imgs = npz_file["test_images"]
            self.labels = npz_file["test_labels"]
        else:
            raise ValueError
        return self.imgs, self.labels

    def __len__(self):
        return self.imgs.shape[0]

    def __repr__(self):
        """Adapted from torchvision.ss"""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info["url"],
                root=self.root,
                filename="{}.npz".format(self.flag),
                md5=self.info["MD5"],
            )
        except:
            raise RuntimeError(
                "Something went wrong when downloading! "
                + "Go to the homepage to download manually. "
                + HOMEPAGE
            )


class MedMNIST2D(MedMNIST_mod):
    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)
        if len(target) == 1:
            target = target[
                0
            ]  # convert from array to value. Might cause errors with some medmnist datasets that are not only multiclass labels
        # if self.as_rgb:
        #    img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def save(self, folder, postfix="png", write_csv=True):
        from medmnist.utils import save2d

        save2d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, self.flag),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}.csv") if write_csv else None,
        )

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(
            imgs=self.imgs, n_channels=self.info["n_channels"], sel=sel
        )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(
                os.path.join(save_folder, f"{self.flag}_{self.split}_montage.jpg")
            )

        return montage_img


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class SuperCIFAR100(datasets.VisionDataset):
    """Super`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This holds out subclasses

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        kwargs: Optional[Callable] = None,
    ) -> None:
        super(SuperCIFAR100, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.coarse_targets = []
        self.fine_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.coarse_targets.extend(entry["coarse_labels"])
                    self.fine_targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        holdout_fine_classes = [
            "seal",
            "flatfish",
            "tulip",
            "bowl",
            "pear",
            "lamp",
            "couch",
            "beetle",
            "lion",
            "skyscraper",
            "sea",
            "cattle",
            "raccoon",
            "lobster",
            "girl",
            "lizard",
            "hamster",
            "oak_tree",
            "motorcycle",
            "tractor",
        ]

        holdout_fine_idx = [self.class_to_idx[cl] for cl in holdout_fine_classes]
        train_data_ix = [
            ix
            for ix, (fine_label, coarse_label) in enumerate(
                zip(self.fine_targets, self.coarse_targets)
            )
            if (fine_label not in holdout_fine_idx and coarse_label != 19)
        ]
        holdout_data_ix = [
            ix
            for ix, (fine_label, coarse_label) in enumerate(
                zip(self.fine_targets, self.coarse_targets)
            )
            if (fine_label in holdout_fine_idx and coarse_label != 19)
        ]

        if self.train:
            self.targets = list(np.array(self.coarse_targets)[train_data_ix])
            self.data = self.data[train_data_ix]
        else:
            self.targets = list(np.array(self.coarse_targets)[holdout_data_ix])
            self.data = self.data[holdout_data_ix]

        num_classes = len(np.unique(self.targets))
        logger.info(
            "SuperCIFAR check num_classes in data {}. Training {}".format(
                num_classes, self.train
            )
        )
        self.classes = self.coarse_classes

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
            self.coarse_classes = data["coarse_label_names"]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            logger.info("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CorruptCIFAR(datasets.VisionDataset):
    """Corrupt`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This adds corruptions

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        kwargs: Optional[Callable] = None,
    ) -> None:
        super(CorruptCIFAR, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.train = False
        base_folder = "CIFAR-C"

        corruptions = [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "shot_noise",
            "snow",
            "zoom_blur",
        ]

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        labels = list(np.load(os.path.join(root, base_folder, "labels.npy")))
        for corr in corruptions:
            file_path = os.path.join(root, base_folder, "{}.npy".format(corr))
            self.data.append(np.load(file_path))
            self.targets.extend(labels)

        self.data = np.vstack(self.data)
        self.classes = eval_utils.cifar100_classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class ImageNet(datasets.ImageNet):
    def __init__(self, root, train, download, transform, kwargs):
        download = None
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class BREEDImageNet(ImageFolder):
    def __init__(self, root, split, download, transform, kwargs):
        target_transform = None
        infor_dir_path = os.path.abspath(os.path.dirname(breeds_hierarchies.__file__))
        ret = make_entity13(infor_dir_path, split="rand")
        superclasses, subclass_split, label_map = ret
        train_subclasses, test_subclasses = subclass_split
        base_path = "ILSVRC/Data/CLS-LOC"
        root = os.path.join(root, base_path, "train")

        if split == "train" or split == "id_test":
            custom_grouping = train_subclasses
        else:
            custom_grouping = test_subclasses

        label_mapping = get_label_mapping("custom_imagenet", custom_grouping)

        super().__init__(
            root,
            loader=None,
            transform=transform,
            target_transform=target_transform,
            label_mapping=label_mapping,
        )

        if split == "train" or split == "id_test":
            rng = np.random.default_rng(12345)
            self.sampels = rng.shuffle(self.samples)
            if split == "train":
                self.samples = self.samples[10000:]
            elif split == "id_test":
                self.samples = self.samples[:10000]

        self.imgs = self.samples
        self.classes = [
            "garment",
            "bird",
            "reptile",
            "arthropod",
            "mammal",
            "accessory",
            "craft",
            "equipment",
            "furniture",
            "instrument",
            "man-made structure",
            "wheeled vehicle",
            "produce",
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class WILDSAnimals(IWildCamDataset):
    def __init__(self, root, train, download, transform):
        super().__init__(
            version=None, root_dir=root, download=True, split_scheme="official"
        )

        logger.debug("CHECK ROOT !!! {}", root)

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """

        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = myWILDSSubset(self, split_idx, transform)
        return subset


class myWILDSSubset(WILDSSubset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices, transform)

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class WILDSCamelyon(Camelyon17Dataset):
    def __init__(self, root, train, download, transform):
        super().__init__(
            version=None, root_dir=root, download=True, split_scheme="official"
        )

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if (
            split == "id_val"
        ):  # shuffle iid set for chamelyon to be able to split into val and test set.
            rng = np.random.default_rng(12345)
            split_idx = rng.permutation(split_idx)
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = myWILDSSubset(self, split_idx, transform)
        return subset


class WILDSAnimalsOpenSet(IWildCamDataset):
    def __init__(
        self, root, train, download, transform, out_classes: list[int] = [0, 1, 2, 3]
    ):
        super().__init__(
            version=None, root_dir=root, download=True, split_scheme="official"
        )
        self.out_classes = out_classes

        logger.debug("CHECK ROOT !!! {}", root)

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """

        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        if split == "train":
            class_mask = ~np.isin(self.y_array, self.out_classes)
            split_idx = np.where(split_mask & class_mask)[0]
        else:
            split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = myWILDSSubset(self, split_idx, transform)
        return subset


class SVHNOpenSet(datasets.SVHN):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        out_classes: list[int] = [0, 1, 2, 3],
    ) -> None:
        super().__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.out_classes = out_classes
        logger.info("SVHN holdout classes {}", self.out_classes)

        if split == "train":
            self.data = self.data[~np.isin(self.labels, self.out_classes)]
            self.labels = self.labels[~np.isin(self.labels, self.out_classes)]
