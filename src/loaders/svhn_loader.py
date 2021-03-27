
from torchvision import datasets
import numpy as np
from src.loaders.abstract_loader import AbstractDataLoader


class DataLoader(AbstractDataLoader):
    def prepare_data(self):
        self.train_dataset = SVHNAlbumentation(
            root=self.data_dir, split="train", download=True, transform=self.augmentations["train"]
        )
        self.val_dataset = SVHNAlbumentation(
            root=self.data_dir, split="train", download=False, transform=self.augmentations["val"]
        )
        self.test_dataset = SVHNAlbumentation(
            root=self.data_dir, split="test", download=True, transform=self.augmentations["test"]
        )


class SVHNAlbumentation(datasets.SVHN):
    def __init__(self, root, split, download, transform):
        super().__init__(root=root, split=split, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = np.transpose(img, (1, 2, 0))

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.target_transform is not None:
            transformed = self.target_transform(target=target)
            target = transformed["target"]

        return img, target

