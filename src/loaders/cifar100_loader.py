
from torchvision import datasets
from src.loaders.abstract_loader import AbstractDataLoader


class DataLoader(AbstractDataLoader):
    def prepare_data(self):
        self.train_dataset = CIFAR100Albumentation(
            root=self.data_dir, train=True, download=True, transform=self.augmentations["train"]
        )
        self.val_dataset = CIFAR100Albumentation(
            root=self.data_dir, train=True, download=False, transform=self.augmentations["val"]
        )
        self.test_dataset = CIFAR100Albumentation(
            root=self.data_dir, train=False, download=True, transform=self.augmentations["test"]
        )


class CIFAR100Albumentation(datasets.CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label





# import matplotlib.pyplot as plt
# image, label = self.data[index], self.targets[index]
# if self.transform is not None:
#     transformed = self.transform(image=image)
#     image = transformed["image"]
# plt.imshow(  image.permute(1, 2, 0)  )
# plt.show()
