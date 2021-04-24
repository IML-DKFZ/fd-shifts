from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

transforms_collection = {
    "compose": lambda x: transforms.Compose(x),
    "to_tensor": transforms.ToTensor,
    "normalize": lambda x: transforms.Normalize(x[0], x[1]),
    "random_crop": lambda x: transforms.RandomCrop(x[0], padding=x[1]),
    "center_crop": lambda x: transforms.CenterCrop(x),
    "randomresized_crop": lambda x: transforms.RandomResizedCrop(x),
    "hflip": lambda x: transforms.RandomHorizontalFlip() if x else None,
    "resize": lambda x: transforms.Resize(x),
    "rotate": lambda x: transforms.RandomRotation(x),
    "color_jitter": lambda x: transforms.ColorJitter(brightness=x[0], contrast=x[1], saturation=x[2]),
    "lighting": lambda x: Lighting()
}
#
# transforms_collection = {
#     "compose": lambda x: A.Compose(x),
#     "to_tensor": ToTensorV2,
#     "normalize": lambda x: A.Normalize(mean=x[0], std=x[1]),
#     "pad": lambda x: A.PadIfNeeded(min_height=x[0], min_width=x[1], border_mode=cv2.BORDER_CONSTANT, value=0),
#     "random_crop": lambda x: A.RandomCrop(height=x[0], width=x[1]),
#     "center_crop": lambda x: A.CenterCrop(height=x[0], width=x[1]),
#     "randomresizecrop": lambda x: A.RandomSizedCrop(min_max_height=x[0], height=x[1], width=x[2], w2h_ratio=1),
#     "flip": lambda x: A.Flip() if x else None,
#     "hflip": lambda x: A.HorizontalFlip() if x else None,
#     "resize": lambda x: A.Resize(height=x[0], width=x[1]),
#     "rotate": lambda x: A.Rotate(limit=x),
#     "color_jitter": lambda x: A.ColorJitter(
#         brightness=x[0], contrast=x[1], saturation=x[2], hue=x[3]
#     ),
# }



class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd=0.05):
        self.alphastd = alphastd

        # IMAGENET PCA
        self.eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])



    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))