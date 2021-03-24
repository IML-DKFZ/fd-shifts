
from torchvision import datasets
from imlone.loaders.old_loader import AbstractDataLoader


class DataLoader(AbstractDataLoader):
    def prepare_data(self):
        self.train_dataset = datasets.SVHN(
            root=self.data_dir, split="train", download=True, transform=self.augmentations["train"]
        )
        self.val_dataset = datasets.SVHN(
            root=self.data_dir, split="train", download=False, transform=self.augmentations["train"]
        )
        self.test_dataset = datasets.SVHN(
            root=self.data_dir, split="test", download=True, transform=None
        )
