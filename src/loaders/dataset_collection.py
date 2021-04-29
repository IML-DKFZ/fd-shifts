
from torchvision import datasets
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from typing import Any, Callable, Optional, Tuple
from robustness.tools.folder import ImageFolder
from robustness.tools.breeds_helpers import make_entity13
from robustness.tools.breeds_helpers import print_dataset_info
from robustness.tools.helpers import get_label_mapping
from robustness.tools.breeds_helpers import ClassHierarchy
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.wilds_dataset import WILDSSubset
from src.loaders import breeds_hierarchies
import numpy as np
from PIL import Image
import io
import pickle
import os


def get_dataset(name, root, train, download, transform, kwargs):
    """
        Return a new instance of dataset loader
    """
    dataset_factory = {
        "svhn": datasets.SVHN,
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "super_cifar100": SuperCIFAR100,
        "corrupt_cifar100": CorruptCIFAR,
        "corrupt_cifar10": CorruptCIFAR,
        "imagenet": BREEDImageNet,
        "wilds_animals": WILDSAnimals,
        "wilds_camelyon": WILDSCamelyon,
    }

    pass_kwargs = {"root": root, "train": train, "download": download, "transform": transform}
    if name == "svhn":
        pass_kwargs = {"root": root, "split": "train" if train else "test", "download": download, "transform": transform}

    if name == "imagenet":
        pass_kwargs["kwargs"] = kwargs

    if not "wilds" in name:
        return dataset_factory[name](**pass_kwargs)
    else:
        return dataset_factory[name](**pass_kwargs).get_subset("train" if train else "test", frac=1.0, transform=transform)

class SuperCIFAR100(datasets.VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

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
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'], ['train', '16019d7e3df5f24257cddd939b257f8d']
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            kwargs: Optional[Callable] = None
    ) -> None:

        super(SuperCIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        self.fine_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['coarse_labels'])
                    self.fine_targets.extend(entry['fine_labels'])

        num_classes = len(np.unique(self.targets))
        print("SuperCIFAR check num_classes in data {}. Training {}".format(num_classes, self.train))
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        holdout_fine_classes = ['seal', 'flatfish', 'tulip', 'bowl', 'pear', 'lamp',
                                'couch', 'beetle', 'lion', 'skyscraper', 'sea', 'cattle', 'raccoon', 'lobster',
                                'girl', 'lizard', 'hamster', 'oak_tree', 'motorcycle', 'tractor']

        holdout_fine_idx = [self.class_to_idx[cl] for cl in holdout_fine_classes]
        train_data_ix = [ix for ix,fine_label in enumerate(self.fine_targets) if fine_label not in holdout_fine_idx]
        holdout_data_ix = [ix for ix, fine_label in enumerate(self.fine_targets) if fine_label in holdout_fine_idx]

        if self.train:
            self.targets = list(np.array(self.targets)[train_data_ix]) # massive speedup compared to list comprehension
            self.data= self.data[train_data_ix]
        else:
            self.targets = list(np.array(self.targets)[holdout_data_ix])
            self.data = self.data[holdout_data_ix]


    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CorruptCIFAR(datasets.VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

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
            kwargs: Optional[Callable] = None
    ) -> None:

        super(CorruptCIFAR, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = False
        base_folder  = "CIFAR-C"

        corruptions = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        # "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        # "saturate",
        "shot_noise",
        "snow",
        # "spatter",
        # "speckle_noise",
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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

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
    def __init__(self, root, train, download, transform, kwargs):
        target_transform = None
        infor_dir_path = os.path.abspath(os.path.dirname(breeds_hierarchies.__file__))
        ret = make_entity13(infor_dir_path, split="rand")
        superclasses, subclass_split, label_map = ret
        train_subclasses, test_subclasses = subclass_split
        base_path = "ILSVRC/Data/CLS-LOC"
        if train:
            root = os.path.join(root, base_path, "train")
            custom_grouping = train_subclasses
        else:
            root = os.path.join(root, base_path, "train")
            custom_grouping = test_subclasses


        #
        # print_dataset_info(superclasses,
        #                    subclass_split,
        #                    label_map,
        #                    ClassHierarchy(kwargs["info_dir_path"]).LEAF_NUM_TO_NAME)

        label_mapping = get_label_mapping("custom_imagenet", custom_grouping)

        super().__init__(root,
                         loader=None,
                         transform=transform,
                         target_transform=target_transform,
                         label_mapping=label_mapping)

        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

#
#
class WILDSAnimals(IWildCamDataset):
    def __init__(self, root, train, download, transform):
        super().__init__(version=None, root_dir=root, download=download, split_scheme='official')

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
        super().__init__(version=None, root_dir=root, download=download, split_scheme='official')

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




# import matplotlib.pyplot as plt
# image, label = self.data[index], self.targets[index]
# if self.transform is not None:
#     transformed = self.transform(image=image)
#     image = transformed["image"]
# plt.imshow(  image.permute(1, 2, 0)  )
# plt.show()

