from typing import Callable, Optional

import numpy as np
from torchvision import datasets

from fd_shifts.loaders.dataset_collection import SVHN, get_dataset


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

        if split == "train":
            self.data = self.data[~np.isin(self.labels, self.out_classes)]
            self.labels = self.labels[~np.isin(self.labels, self.out_classes)]


def test_svhn():
    s1 = datasets.SVHN(root="/home/t974t/Data/svhn", split="train", download=True)
    s2 = SVHN(root="/home/t974t/Data/svhn", train=True, split="all", download=True)

    assert len(s1) == len(s2)
    for i in range(len(s1)):
        assert s1[i] == s2[i]

    s1 = datasets.SVHN(root="/home/t974t/Data/svhn", split="test", download=True)
    s2 = SVHN(root="/home/t974t/Data/svhn", train=False, split="all", download=True)

    assert len(s1) == len(s2)
    for i in range(len(s1)):
        assert s1[i] == s2[i]

    s1 = SVHNOpenSet(
        root="/home/t974t/Data/svhn",
        split="test",
        download=True,
        out_classes=[0, 1, 3, 8, 2],
    )
    s2 = SVHN(
        root="/home/t974t/Data/svhn",
        train=False,
        split="openset",
        download=True,
        out_classes=[0, 1, 3, 8, 2],
    )

    assert len(s1) == len(s2)
    for i in range(len(s1)):
        assert s1[i] == s2[i]

    s1 = get_dataset(
        name="svhn",
        root="/home/t974t/Data/svhn",
        train=True,
        download=True,
        transform=None,
        target_transform=None,
        kwargs={"out_classes": [0, 1, 3, 8, 2]},
    )
    s2 = datasets.SVHN(root="/home/t974t/Data/svhn", split="train", download=True)
    assert len(s1) == len(s2)
    for i in range(len(s1)):
        assert s1[i] == s2[i]

    s1 = get_dataset(
        name="svhn",
        root="/home/t974t/Data/svhn",
        train=False,
        download=True,
        transform=None,
        target_transform=None,
        kwargs={"out_classes": [0, 1, 3, 8, 2]},
    )
    s2 = datasets.SVHN(root="/home/t974t/Data/svhn", split="test", download=True)
    assert len(s1) == len(s2)
    for i in range(len(s1)):
        assert s1[i] == s2[i]

    s1 = SVHNOpenSet(
        root="/home/t974t/Data/svhn",
        split="test",
        download=True,
        out_classes=[0, 1, 3, 8, 2],
    )
    s2 = get_dataset(
        name="svhn_openset",
        root="/home/t974t/Data/svhn",
        train=False,
        download=True,
        transform=None,
        target_transform=None,
        kwargs={"out_classes": [0, 1, 3, 8, 2]},
    )
    assert len(s1) == len(s2)
    for i in range(len(s1)):
        assert s1[i] == s2[i]
