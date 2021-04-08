
from torchvision import datasets
import numpy as np
from src.loaders.abstract_loader import AbstractDataLoader

class DataLoader(AbstractDataLoader):

    def prepare_data(self):
        print("looking for dataset at", self.data_dir)
        self.train_dataset = MNISTAlbumentation(
            root=self.data_dir, train=True, download=False, transform=self.augmentations["train"]
        )
        self.val_dataset = MNISTAlbumentation(
            root=self.data_dir, train=True, download=False, transform=self.augmentations["val"]
        )
        self.test_dataset = MNISTAlbumentation(
            root=self.data_dir, train=False, download=False, transform=self.augmentations["test"]
        )


class MNISTAlbumentation(datasets.MNIST):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img = np.transpose(img, (1, 2, 0))
        if self.transform is not None:
            transformed = self.transform(image=img.numpy())
            img = transformed["image"]

        if self.target_transform is not None:
            transformed = self.target_transform(target=target.numpy())
            target = transformed["target"]

        return img, target

