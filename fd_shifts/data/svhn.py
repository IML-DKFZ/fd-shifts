from typing import Callable, Literal, Optional

import numpy as np
from torchvision import datasets


class SVHN(datasets.SVHN):
    """SVHN dataset with support for Open Set splits.

    Attributes:
        out_classes: Classes to exclude from the training set.
        train: Whether to load the training or test split.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        split: Literal["all", "openset"] = "all",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        out_classes: list[int] = [0, 1, 2, 3],
    ):
        super().__init__(
            root,
            split="train" if train else "test",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.out_classes = out_classes
        self.train = train
        if split == "openset" and train:
            self.data = self.data[~np.isin(self.labels, self.out_classes)]
            self.labels = self.labels[~np.isin(self.labels, self.out_classes)]
