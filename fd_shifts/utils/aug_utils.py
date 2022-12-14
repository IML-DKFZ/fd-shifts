from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np

transforms_collection = {
    "compose": lambda x: transforms.Compose(x),
    "to_tensor": transforms.ToTensor(),
    "normalize": lambda x: transforms.Normalize(x[0], x[1]),
    "random_crop": lambda x: transforms.RandomCrop(x[0], padding=x[1]),
    "center_crop": lambda x: transforms.CenterCrop(x),
    "randomresized_crop": lambda x: transforms.RandomResizedCrop(x),
    "hflip": lambda x: transforms.RandomHorizontalFlip() if x else None,
    "resize": lambda x: transforms.Resize(size=x),
    "rotate": lambda x: transforms.RandomRotation(x),
    "color_jitter": lambda x: transforms.ColorJitter(
        brightness=x[0], contrast=x[1], saturation=x[2]
    ),
    "lighting": lambda x: Lighting(),
    "cutout": lambda x: Cutout(length=x),
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """

    def __init__(self, alphastd=0.05):
        self.alphastd = alphastd

        # IMAGENET PCA
        self.eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = torch.Tensor(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Cutout(object):
    """Randomly mask out one or more patches from an image.
       https://arxiv.org/abs/1708.04552
    Args:
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if np.random.choice([0, 1]):
            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img
