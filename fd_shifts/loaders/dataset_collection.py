import imghdr
import io
import os
import pickle
from typing import Any, Callable, Optional, Tuple, TypeVar

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from medmnist.info import DEFAULT_ROOT, HOMEPAGE, INFO
from PIL import Image, ImageFile
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


def get_df(out_dim, data_dir, data_folder):
    # 2020 data
    df_train = pd.read_csv(
        os.path.join(
            data_dir, f"jpeg-melanoma-{data_folder}x{data_folder}", "train.csv"
        )
    )
    df_train = df_train[df_train["tfrecord"] != -1].reset_index(drop=True)
    df_train["filepath"] = df_train["image_name"].apply(
        lambda x: os.path.join(
            data_dir, f"jpeg-melanoma-{data_folder}x{data_folder}/train", f"{x}.jpg"
        )
    )

    df_train["is_ext"] = 0

    # 2018, 2019 data (external data)
    df_train2 = pd.read_csv(
        os.path.join(
            data_dir, f"jpeg-isic2019-{data_folder}x{data_folder}", "train.csv"
        )
    )
    df_train2 = df_train2[df_train2["tfrecord"] >= 0].reset_index(drop=True)
    df_train2["filepath"] = df_train2["image_name"].apply(
        lambda x: os.path.join(
            data_dir, f"jpeg-isic2019-{data_folder}x{data_folder}/train", f"{x}.jpg"
        )
    )

    df_train2["fold"] = df_train2["tfrecord"] % 5
    df_train2["is_ext"] = 1

    # Preprocess Target
    df_train["diagnosis"] = df_train["diagnosis"].apply(
        lambda x: x.replace("seborrheic keratosis", "BKL")
    )
    df_train["diagnosis"] = df_train["diagnosis"].apply(
        lambda x: x.replace("lichenoid keratosis", "BKL")
    )
    df_train["diagnosis"] = df_train["diagnosis"].apply(
        lambda x: x.replace("solar lentigo", "BKL")
    )
    df_train["diagnosis"] = df_train["diagnosis"].apply(
        lambda x: x.replace("lentigo NOS", "BKL")
    )
    df_train["diagnosis"] = df_train["diagnosis"].apply(
        lambda x: x.replace("cafe-au-lait macule", "unknown")
    )
    df_train["diagnosis"] = df_train["diagnosis"].apply(
        lambda x: x.replace("atypical melanocytic proliferation", "unknown")
    )

    if out_dim == 9:
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("NV", "nevus")
        )
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("MEL", "melanoma")
        )
    elif out_dim == 4:
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("NV", "nevus")
        )
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("MEL", "melanoma")
        )
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("DF", "unknown")
        )
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("AK", "unknown")
        )
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("SCC", "unknown")
        )
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("VASC", "unknown")
        )
        df_train2["diagnosis"] = df_train2["diagnosis"].apply(
            lambda x: x.replace("BCC", "unknown")
        )
    else:
        raise NotImplementedError()

    # concat train data
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

    # test data
    df_test = pd.read_csv(
        os.path.join(data_dir, f"jpeg-melanoma-{data_folder}x{data_folder}", "test.csv")
    )
    df_test["filepath"] = df_test["image_name"].apply(
        lambda x: os.path.join(
            data_dir, f"jpeg-melanoma-{data_folder}x{data_folder}/test", f"{x}.jpg"
        )
    )

    meta_features = None
    n_meta_features = 0

    # class mapping
    diagnosis2idx = {
        d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))
    }
    df_train["target"] = df_train["diagnosis"].map(diagnosis2idx)
    mel_idx = diagnosis2idx["melanoma"]

    return df_train, df_test, meta_features, n_meta_features, mel_idx


def get_transforms(image_size):
    transforms_train = albumentations.Compose(
        [
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.OneOf(
                [
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ],
                p=0.7,
            ),
            albumentations.OneOf(
                [
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(num_steps=5, distort_limit=1.0),
                    albumentations.ElasticTransform(alpha=3),
                ],
                p=0.7,
            ),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85
            ),
            albumentations.Resize(image_size, image_size),
            albumentations.Cutout(
                max_h_size=int(image_size * 0.375),
                max_w_size=int(image_size * 0.375),
                num_holes=1,
                p=0.7,
            ),
            albumentations.Normalize(),
        ]
    )

    transforms_val = albumentations.Compose(
        [albumentations.Resize(image_size, image_size), albumentations.Normalize()]
    )

    return transforms_train, transforms_val


class MelanomaDataset(Dataset):
    def __init__(self, csv: pd.DataFrame, train: bool, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Rxrx1Dataset(Dataset):
    """
    Returns 6-Channel image, not rgb but stacked greychannels from fluoresenzemicroscopy
    """

    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.targets = self.csv.target
        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        filepath = row.stempath
        channels = []
        for channel in range(1, 7, 1):
            start, end = filepath.split("XXX")
            channel_path = start + str(channel) + end
            channels.append(channel_path)
        blue = cv2.imread(channels[0])[:, :, 0]
        green = cv2.imread(channels[1])[:, :, 0]
        red = cv2.imread(channels[2])[:, :, 0]
        cyan = cv2.imread(channels[3])[:, :, 0]
        magenta = cv2.imread(channels[4])[:, :, 0]
        yellow = cv2.imread(channels[5])[:, :, 0]

        image = np.stack((red, green, blue, cyan, magenta, yellow), axis=2)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.astype(np.float32)
        data = image

        return data, torch.tensor(self.csv.iloc[index].target).long()


class XrayDataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.targets = self.csv.target.to_list()
        self.imgs = self.csv["filepath"].to_list()
        self.samples = self.imgs

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = image.astype(np.float32)
        data = image

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Lidc_idriDataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.targets = self.csv.target.to_list()
        self.imgs = self.csv["filepath"].to_list()
        self.samples = self.imgs

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = image.astype(np.float32)
        data = image

        return data, torch.tensor(self.csv.iloc[index].target).long()


class BasicDataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target
        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class DermoscopyAllDataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
        oversampeling: int = 0,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv
        if self.train:
            # oversample malignant class for ~50/50 ratio
            if oversampeling > 0:
                df_mal = self.train_df[self.train_df.target == 1]
                for _ in range(oversampeling):
                    self.train_df = pd.concat(
                        [
                            self.train_df,
                            df_mal,
                        ]
                    )
            self.csv = self.train_df

        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class D7pDataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Ham10000Dataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Ham10000DatasetSubbig(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Ham10000DatasetSubsmall(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Isic2020Dataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Ph2Dataset(Dataset):
    def __init__(
        self,
        csv: pd.DataFrame,
        train: bool,
        transform: Optional[Callable] = None,
    ):
        self.csv = csv.reset_index(drop=True)
        self.train = train
        self.transform = transform
        self.train_df = self.csv.sample(frac=0.8, random_state=200)
        self.test_df = self.csv.drop(self.train_df.index)
        if self.train:
            self.csv = self.train_df
        elif not self.train:
            self.csv = self.test_df
        self.targets = self.csv.target

        self.imgs = self.csv["filepath"]
        self.samples = self.imgs

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        return data, torch.tensor(self.csv.iloc[index].target).long()


class Isicv01(Dataset):
    "Class with binary classification benign vs malignant of skin cancer and control"

    def __init__(
        self,
        csv_file: str,
        root: str,
        transform: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
        train: bool = True,
        download: bool = False,
    ):
        """
        Args:
            csv_file (string): Path to csv with metadata and images
            root_dir (string): Directory with the images
            transforms (Callable, optional): Torchvision transforms to apply
            target_transforms (Callable): target transforms to apply
            train (bool): If true traindata if false test data
            download (bool): toDo
        """
        self.isicv01_df = pd.read_csv(csv_file)

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.download = download

        self.target_transforms = target_transforms
        self.transforms = transform
        self.train = train
        self.resample_malignant: int = 4
        self.data, self.targets = self._load_data()
        self.classes = {"bening": 0, "malignant": 1}

    def __len__(self):
        return len(self.targets)

    def _load_data(self) -> Tuple[Any, Any]:
        self.train_df = self.isicv01_df.sample(frac=0.8, random_state=200)
        self.test_df = self.isicv01_df.drop(self.train_df.index)
        if self.resample_malignant > 0:
            mal = self.train_df["class"] == 1
            mal_df = self.train_df[mal]
            self.train_df = self.train_df.append(
                [mal_df] * self.resample_malignant, ignore_index=True
            )
        if self.train:
            image_files = self.train_df["isic_id"]
            target_series = self.train_df["class"]
        elif not self.train:
            image_files = self.test_df["isic_id"]
            target_series = self.test_df["class"]
        img_path = self.root + image_files + ".jpg"
        data = []
        target = []
        for x in range(len(image_files)):
            target.append(target_series.iloc[x])
            image = cv2.imread(img_path.iloc[x])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data.append(image)
            # data.append(Image.open(img_path.iloc[x]))
        return data, target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        """
        Args:
            index (int): image and label index in the dataframe to return
        Returns:
            tupel (image, target): where target is the label of the target class
        """
        img, target = self.data[index], int(self.targets[index])
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target


class MedMNIST_mod(Dataset):
    flag = ...

    def __init__(
        self,
        root: str,
        split,
        transform=None,
        target_transform=None,
        download=False,
        as_rgb=False,
    ):
        """dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        """

        self.info = INFO[self.flag]
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        # root = os.path.expanduser(root)  # recognize ~ as home directory
        # if root is not None and os.path.exists(root):
        #     self.root = root
        # else:
        #     raise RuntimeError(
        #         "Failed to setup the default `root` directory. "
        #         + "Please specify and create the `root` directory manually."
        #     )

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
        # if len(target) == 1:
        #     target = target[
        #         0
        #     ]  # convert from array to value. Might cause errors with some medmnist datasets that are not only multiclass labels
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


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class SuperCIFAR100(datasets.VisionDataset):
    """Super`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This holds out subclasses

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (Callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (Callable, optional): A function/transform that takes in the
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
        transform (Callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (Callable, optional): A function/transform that takes in the
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
    "med_mnist_oct": OCTMNIST,
    "med_mnist_pneu": PneumoniaMNIST,
    "med_mnist_chest": ChestMNIST,
    "med_mnist_derma": DermaMNIST,
    "med_mnist_retina": RetinaMNIST,
    "med_mnist_breast": BreastMNIST,
    "med_mnist_blood": BloodMNIST,
    "med_mnist_tissue": TissueMNIST,
    "med_mnist_organ_a": OrganAMNIST,
    "xray_chestall": XrayDataset,
    "xray_chestallnih14": XrayDataset,
    "xray_chestallchexpert": XrayDataset,
    "xray_chestallmimic": XrayDataset,
    "xray_chestallbutnih14": XrayDataset,
    "xray_chestallbutchexpert": XrayDataset,
    "xray_chestallbutmimic": XrayDataset,
    "xray_chestallcorrletter": XrayDataset,
    "xray_chestallcorrbrlow": XrayDataset,
    "xray_chestallcorrbrlowlow": XrayDataset,
    "xray_chestallcorrbrhigh": XrayDataset,
    "xray_chestallcorrbrhighhigh": XrayDataset,
    "xray_chestallcorrmotblrhigh": XrayDataset,
    "xray_chestallcorrmotblrhighhigh": XrayDataset,
    "xray_chestallcorrgaunoilow": XrayDataset,
    "xray_chestallcorrgaunoilowlow": XrayDataset,
    "xray_chestallcorrelastichigh": XrayDataset,
    "xray_chestallcorrelastichighhigh": XrayDataset,
    "rxrx1all": Rxrx1Dataset,
    "rxrx1all_buthepg2": Rxrx1Dataset,
    "rxrx1all_buthuvec": Rxrx1Dataset,
    "rxrx1all_butu2os": Rxrx1Dataset,
    "rxrx1all_butrpe": Rxrx1Dataset,
    "rxrx1all_onlyhepg2": Rxrx1Dataset,
    "rxrx1all_onlyhuvec": Rxrx1Dataset,
    "rxrx1all_onlyu2os": Rxrx1Dataset,
    "rxrx1all_onlyrpe": Rxrx1Dataset,
    "rxrx1all_large_set1": Rxrx1Dataset,
    "rxrx1all_large_set2": Rxrx1Dataset,
    "rxrx1all_large_set3": Rxrx1Dataset,
    "rxrx1all_large_set4": Rxrx1Dataset,
    "rxrx1all_large_set5": Rxrx1Dataset,
    "rxrx1all_small_set1": Rxrx1Dataset,
    "rxrx1all_small_set2": Rxrx1Dataset,
    "rxrx1all_small_set3": Rxrx1Dataset,
    "rxrx1all_small_set4": Rxrx1Dataset,
    "rxrx1all_small_set5": Rxrx1Dataset,
    "rxrx1allcorrbrlow": Rxrx1Dataset,
    "rxrx1allcorrbrlowlow": Rxrx1Dataset,
    "rxrx1allcorrbrhigh": Rxrx1Dataset,
    "rxrx1allcorrbrhighhigh": Rxrx1Dataset,
    "rxrx1allcorrmotblrhigh": Rxrx1Dataset,
    "rxrx1allcorrmotblrhighhigh": Rxrx1Dataset,
    "rxrx1allcorrgaunoilow": Rxrx1Dataset,
    "rxrx1allcorrgaunoilowlow": Rxrx1Dataset,
    "rxrx1allcorrelastichigh": Rxrx1Dataset,
    "rxrx1allcorrelastichighhigh": Rxrx1Dataset,
    "lidc_idriall": Lidc_idriDataset,
    "lidc_idriallcorrbrlow": Lidc_idriDataset,
    "lidc_idriallcorrbrlowlow": Lidc_idriDataset,
    "lidc_idriallcorrbrhigh": Lidc_idriDataset,
    "lidc_idriallcorrbrhighhigh": Lidc_idriDataset,
    "lidc_idriallcorrmotblrhigh": Lidc_idriDataset,
    "lidc_idriallcorrmotblrhighhigh": Lidc_idriDataset,
    "lidc_idriallcorrgaunoilow": Lidc_idriDataset,
    "lidc_idriallcorrgaunoilowlow": Lidc_idriDataset,
    "lidc_idriallcorrelastichigh": Lidc_idriDataset,
    "lidc_idriallcorrelastichighhigh": Lidc_idriDataset,
    "lidc_idriall_calcification_iid": Lidc_idriDataset,
    "lidc_idriall_calcification_ood": Lidc_idriDataset,
    "lidc_idriall_spiculation_iid": Lidc_idriDataset,
    "lidc_idriall_spiculation_ood": Lidc_idriDataset,
    "lidc_idriall_texture_iid": Lidc_idriDataset,
    "lidc_idriall_texture_ood": Lidc_idriDataset,
    "isic_v01": Isicv01,
    "isic_v01_cr": Isicv01,
    "isic_winner": MelanomaDataset,
    "dermoscopyall": DermoscopyAllDataset,
    "dermoscopyalld7p": DermoscopyAllDataset,
    "dermoscopyallph2": DermoscopyAllDataset,
    "dermoscopyallbarcelona": DermoscopyAllDataset,
    "dermoscopyallqueensland": DermoscopyAllDataset,
    "dermoscopyallvienna": DermoscopyAllDataset,
    "dermoscopyallmskcc": DermoscopyAllDataset,
    "dermoscopyallpascal": DermoscopyAllDataset,
    "dermoscopyallbutd7p": DermoscopyAllDataset,
    "dermoscopyallbutph2": DermoscopyAllDataset,
    "dermoscopyallbutbarcelona": DermoscopyAllDataset,
    "dermoscopyallbutqueensland": DermoscopyAllDataset,
    "dermoscopyallbutvienna": DermoscopyAllDataset,
    "dermoscopyallbutmskcc": DermoscopyAllDataset,
    "dermoscopyallbutpascal": DermoscopyAllDataset,
    "dermoscopyallcorrbrlow": DermoscopyAllDataset,
    "dermoscopyallcorrbrlowlow": DermoscopyAllDataset,
    "dermoscopyallcorrbrhigh": DermoscopyAllDataset,
    "dermoscopyallcorrbrhighhigh": DermoscopyAllDataset,
    "dermoscopyallcorrmotblrhigh": DermoscopyAllDataset,
    "dermoscopyallcorrmotblrhighhigh": DermoscopyAllDataset,
    "dermoscopyallcorrgaunoilow": DermoscopyAllDataset,
    "dermoscopyallcorrgaunoilowlow": DermoscopyAllDataset,
    "dermoscopyallcorrgaunoihigh": DermoscopyAllDataset,
    "dermoscopyallcorrgaunoihighhigh": DermoscopyAllDataset,
    "dermoscopyallcorrelastichigh": DermoscopyAllDataset,
    "dermoscopyallcorrelastichighhigh": DermoscopyAllDataset,
    "dermoscopy_isic_2020": DermoscopyAllDataset,
    "ph2": Ph2Dataset,
    "d7p": D7pDataset,
    "dermoscopyallham10000multi": DermoscopyAllDataset,
    "dermoscopyallham10000subbig": DermoscopyAllDataset,
    "dermoscopyallham10000subsmall": DermoscopyAllDataset,
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


def dataset_exists(name: str) -> bool:
    """Check if dataset with name is registered

    Args:
        name (str): name of the dataset

    Returns:
        True if it exists
    """
    return name in _dataset_factory


def register_dataset(name: str, dataset: type) -> None:
    """Register a new dataset

    Args:
        name (str): name to register under
        dataset (type): dataset class to register
    """
    _dataset_factory[name] = dataset


def get_dataset(
    name: str,
    root: str,
    train: bool,
    download: bool,
    transform: Callable,
    target_transform: Callable | None,
    kwargs: dict[str, Any],
) -> Any:
    """Return a new instance of a dataset

    Args:
        name (str): name of the dataset
        root (str): where it is stored on disk
        train (bool): whether to load the train split
        download (bool): whether to attempt to download if it is not in root
        transform (Callable): transforms to apply to loaded data
        target_transform (Callable): transforms to apply to loaded targets
        kwargs (dict[str, Any]): other kwargs to pass on

    Returns:
        dataset instance
    """
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
            "root": root,
            "split": "train" if train else "test",
            "download": download,
            "transform": transform,
            "target_transform": target_transform,
        }
    if "oct" in name:
        pass_kwargs = {
            "root": root,
            "split": "train" if train else "val",  # test set very small. Workaround
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

    elif name == "isicv01":
        pass_kwargs = {
            "root": root,
            "train": train,
            "download": download,
            "transform": transform,
            "csv_file": "/home/l049e/Projects/ISIC/isic_v01_dataframe.csv",
        }
        return _dataset_factory[name](**pass_kwargs)
    elif name == "isic_v01_cr":
        pass_kwargs = {
            "root": root,
            "train": train,
            "download": download,
            "transform": transform,
            "csv_file": "/home/l049e/Projects/ISIC/isic_v01_dataframe.csv",
        }
        return _dataset_factory[name](**pass_kwargs)

    elif name == "isic_winner":
        out_dim = 9
        data_dir = root
        data_folder = "512"  # input image size
        df_train, df_test, meta_features, n_meta_features, mel_idx = get_df(
            out_dim, data_dir, data_folder
        )
        transforms_train, transforms_val = get_transforms(512)
        if train:
            transforms = transforms_train
        else:
            transforms = transforms_val
        pass_kwargs = {"csv": df_train, "train": train, "transform": transforms}
        return _dataset_factory[name](**pass_kwargs)

    elif name == "d7p":
        out_dim = 2
        data_dir = root
        data_folder = "512"  # input image size
        csv_file = f"{root}/d7p_binaryclass"
        df_train = pd.read_csv(csv_file)
        df_train["filepath"] = root + "/" + df_train["filepath"]
        transforms_train, transforms_val = get_transforms(512)
        if train:
            transforms = transforms_train
        else:
            transforms = transforms_val
        pass_kwargs = {"csv": df_train, "train": train, "transform": transforms}
        return _dataset_factory[name](**pass_kwargs)

    elif name == "isic_2020":
        out_dim = 2
        data_dir = root
        data_folder = "512"  # input image size
        csv_file = f"{root}/isic2020_binaryclass"
        df_train = pd.read_csv(csv_file)
        df_train["filepath"] = root + "/" + df_train["filepath"]

        transforms_train, transforms_val = get_transforms(512)
        if train:
            transforms = transforms_train
        else:
            transforms = transforms_val
        pass_kwargs = {"csv": df_train, "train": train, "transform": transforms}
        return _dataset_factory[name](**pass_kwargs)
    elif name == "ph2":
        out_dim = 2
        data_dir = root
        data_folder = "512"  # input image size
        csv_file = f"{root}/ph2_binaryclass"
        df_train = pd.read_csv(csv_file)
        df_train["filepath"] = root + "/" + df_train["filepath"]
        transforms_train, transforms_val = get_transforms(512)
        if train:
            transforms = transforms_train
        else:
            transforms = transforms_val
        pass_kwargs = {"csv": df_train, "train": train, "transform": transforms}
        return _dataset_factory[name](**pass_kwargs)
    elif "dermoscopyall" in name:
        oversampeling = 0
        binary = "binaryclass"
        if name == "dermoscopyall" or "corr" in name:
            dataset_name = "all"
        else:
            _, dataset_name = name.split("dermoscopyall")
        if dataset_name in ["d7p", "ph2"]:
            dataset = dataset_name
        else:
            dataset = "isic_2020"

        if train:
            mode = "train"
        else:
            mode = "test"
        if dataset_name in ["d7p", "ph2", "pascal"]:
            mode = "test"
        if "multi" in name:
            dataset = "ham10000"
            dataset_name = "ham10000"
            binary = "multiclass"
        if name == "dermoscopyallham10000subbig":
            dataset = "ham10000"
            dataset_name = "ham10000"
            binary = "subbig"
        if name == "dermoscopyallham10000subsmall":
            dataset = "ham10000"
            dataset_name = "ham10000"
            binary = "subsmall"
            mode = "test"

        dataroot = os.environ["DATASET_ROOT_DIR"]
        csv_file = f"{dataroot}/{dataset}/{dataset_name}_{binary}_{mode}.csv"
        df_train = pd.read_csv(csv_file)

        for i in range(len(df_train)):
            start, end = df_train["filepath"].iloc[i].split(".")
            atti = df_train["attribution"].iloc[i]
            if atti in ["ham10000", "d7p", "ph2"]:
                dataset = atti
            else:
                dataset = "isic_2020"
            datafolder = "/" + dataset
            data_dir = os.path.join(dataroot + datafolder)

            # create new path for corrupted images
            if "corr" in name:
                _, cor = name.split("dermoscopyallcorr")
                cor = "_" + cor
            else:
                cor = ""
            df_train.iloc[i, df_train.columns.get_loc("filepath")] = (
                data_dir + "/" + start + "_512" + cor + "." + end
            )

        transforms_train, transforms_val = get_transforms(512)
        if train:
            transforms = transforms_train
        else:
            transforms = transforms_val
        if oversampeling is None:
            oversampeling = 0
        ## for the small ph2 set use train set also for testing. otherwise only 40 images

        pass_kwargs = {
            "csv": df_train,
            "train": train,
            "transform": transforms,
            "oversampeling": oversampeling,
        }
        return _dataset_factory[name](**pass_kwargs)

    elif "xray_chest" in name:
        binary = "multiclass"
        _, dataset_name = name.split("xray_chestall")

        if "but" in name:
            _, dataset = dataset_name.split("but")
        else:
            dataset = dataset_name
        if train:
            mode = "train"
        else:
            mode = "test"

        if name == "xray_chestall" or "corr" in name:
            dataset = "mimic"
            dataset_name = "all"
        dataroot = os.environ["DATASET_ROOT_DIR"]
        csv_file = f"{dataroot}/{dataset}/{dataset_name}_{binary}_{mode}.csv"
        df = pd.read_csv(csv_file)

        for i in range(len(df)):
            atti = df["attribution"].iloc[i]
            dataset = atti
            datafolder = "/" + dataset
            data_dir = os.path.join(dataroot + datafolder)
            img_sub_path = df["filepath"].iloc[i]
            img_path = data_dir + "/" + img_sub_path
            if ".png" in img_path:
                start, _ = img_path.split(".png")
                end = "png"
            if ".jpg" in img_path:
                start, _ = img_path.split(".jpg")
                end = "jpg"

            # create new path for corrupted images
            if "corr" in name:
                _, cor = name.split("xray_chestallcorr")
                cor = "_" + cor
            else:
                cor = ""
            df.iloc[i, df.columns.get_loc("filepath")] = (
                start + "_256" + cor + "." + end
            )

        pass_kwargs = {"csv": df, "train": train, "transform": transform}
        return _dataset_factory[name](**pass_kwargs)

    elif "rxrx1" in name:
        dataroot = os.environ["DATASET_ROOT_DIR"]
        dataset = "rxrx1"
        if train:
            mode = "train"
        else:
            mode = "test"
        if name == "rxrx1all" or "corr" in name:
            if train:
                mode = "train"
            else:
                mode = "test"

            csv_file = f"{dataroot}/{dataset}/{dataset}_multiclass_all_{mode}.csv"
            df = pd.read_csv(csv_file)

        if "but" in name:
            _, cell = name.split("but")
            if train:
                mode = "train"
            else:
                mode = "test"
            df = pd.read_csv(
                f"{dataroot}/{dataset}/{dataset}_multiclass_but_{cell}_{mode}.csv"
            )
        elif "only" in name:
            _, cell = name.split("only")

            mode = "test"
            df = pd.read_csv(
                f"{dataroot}/{dataset}/{dataset}_multiclass_only_{cell}_{mode}.csv"
            )
        elif "set" in name:
            _, set_id = name.split("set")
            if "large" in name:
                largeOrSmall = "large"
            elif "small" in name:
                largeOrSmall = "small"
                mode = "test"

            df = pd.read_csv(
                f"{dataroot}/{dataset}/{dataset}_multiclass_{largeOrSmall}_set{set_id}_{mode}.csv"
            )
        df["filepath"] = dataroot + "/" + dataset + "/" + df["filepath"]
        datafolder = "/" + dataset
        data_dir = os.path.join(dataroot + datafolder)

        for i in range(len(df)):
            img_sub_path = df["stempath"].iloc[i]
            img_path = data_dir + "/" + img_sub_path
            start, _ = img_path.split(".png")
            end = "png"
            if "corr" in name:
                _, cor = name.split("rxrx1allcorr")
                cor = "_" + cor
            else:
                cor = ""
            df.iloc[i, df.columns.get_loc("stempath")] = start + cor + "." + end

        # if name == "rxrx1all_3cell":
        #     df = df[df["cell_type"] != "U2OS"]
        # elif name == "rxrx1all_1cell":
        #     df = df[df["cell_type"] == "U2OS"]
        # elif name == "rxrx1all_40s":
        #     df = df[
        #         ~(
        #             (df["experiment"] == "HEPG2-08")
        #             | (df["experiment"] == "HEPG2-09")
        #             | (df["experiment"] == "HEPG2-11")
        #             | (df["experiment"] == "HEPG2-07")
        #             | (df["experiment"] == "HUVEC-18")
        #             | (df["experiment"] == "HUVEC-19")
        #             | (df["experiment"] == "HUVEC-20")
        #             | (df["experiment"] == "RPE-08")
        #             | (df["experiment"] == "RPE-09")
        #             | (df["experiment"] == "U2OS-01")
        #             | (df["experiment"] == "HUVEC-13")
        #         )
        #     ]
        # elif name == "rxrx1all_11s":
        #     df = df[
        #         (df["experiment"] == "HEPG2-08")
        #         | (df["experiment"] == "HEPG2-09")
        #         | (df["experiment"] == "HEPG2-11")
        #         | (df["experiment"] == "HEPG2-07")
        #         | (df["experiment"] == "HUVEC-18")
        #         | (df["experiment"] == "HUVEC-19")
        #         | (df["experiment"] == "HUVEC-20")
        #         | (df["experiment"] == "RPE-08")
        #         | (df["experiment"] == "RPE-09")
        #         | (df["experiment"] == "U2OS-01")
        #         | (df["experiment"] == "HUVEC-13")
        #     ]

        pass_kwargs = {"csv": df, "train": train, "transform": transform}
        return _dataset_factory[name](**pass_kwargs)

    elif "lidc_idri" in name:
        dataroot = os.environ["DATASET_ROOT_DIR"]
        dataset = "lidc_idri"
        iidOrood = "iid"

        shift = "all"
        if train:
            mode = "train"
        else:
            mode = "test"
        if "calcification" in name:
            shift = "calcification"
        elif "spiculation" in name:
            shift = "spiculation"
        elif "texture" in name:
            shift = "texture"

        if "ood" in name:
            iidOrood = "ood"
            mode = "test"

        csv_file = (
            f"{dataroot}/{dataset}/{dataset}_binaryclass_{shift}_{iidOrood}_{mode}.csv"
        )
        if name == "lidc_idriall" or "corr" in name:
            csv_file = f"{dataroot}/{dataset}/{dataset}_binaryclass_{shift}_{mode}.csv"

        df = pd.read_csv(csv_file)
        # df["filepath"] = dataroot + "/" + df["filepath"]
        datafolder = "/" + dataset
        data_dir = os.path.join(dataroot + datafolder)
        for i in range(len(df)):
            img_sub_path = df["filepath"].iloc[i]
            img_path = data_dir + "/" + img_sub_path
            start, _ = img_path.split(".png")
            end = "png"
            if "corr" in name:
                _, cor = name.split("lidc_idriallcorr")
                cor = "_" + cor
            else:
                cor = ""
            df.iloc[i, df.columns.get_loc("filepath")] = start + cor + "." + end
        # to make iid testset and corruption sets the same images (and we use part of iid for val)
        # images used in val need to be removed from corr. This is necessary here because of small set sizes
        # I assume the "tenPercent" split from abstract dataloader
        # but there is currently a bug where the iid test is [:-split] in the raw outputs
        # so basically the last images are never used neither in val
        # not in iid. Because I can not figure it out currently the corruptions are adjusted to this.
        if "corr" in name:
            length_test = len(df)
            split = int(length_test * 0.1)
            df = df.iloc[:-split]
        pass_kwargs = {"csv": df, "train": train, "transform": transform}
        return _dataset_factory[name](**pass_kwargs)

    else:
        return _dataset_factory[name](**pass_kwargs)
