from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# transforms_collection = {
#     "compose": lambda x: transforms.Compose(x),
#     "to_tensor": transforms.ToTensor,
#     "normalize": lambda x: transforms.Normalize(x[0], x[1]),
#     "random_crop": lambda x: transforms.RandomCrop(x, padding=4),
#     "center_crop": lambda x: transforms.CenterCrop(x),
#     "hflip": lambda x: transforms.RandomHorizontalFlip() if x else None,
#     "resize": lambda x: transforms.Resize(x),
#     "rotate": lambda x: transforms.RandomRotation(x),
#     "color_jitter": lambda x: transforms.ColorJitter(
#         brightness=x[0], contrast=x[1], saturation=x[2], hue=x[3]
#     ),
# }

transforms_collection = {
    "compose": lambda x: A.Compose(x),
    "to_tensor": ToTensorV2,
    "normalize": lambda x: A.Normalize(mean=x[0], std=x[1]),
    "pad": lambda x: A.PadIfNeeded(min_height=x[0], min_width=x[1], border_mode=cv2.BORDER_CONSTANT, value=0),
    "random_crop": lambda x: A.RandomCrop(height=x[0], width=x[1]),
    "center_crop": lambda x: A.CenterCrop(height=x[0], width=x[1]),
    "flip": lambda x: A.Flip() if x else None,
    "hflip": lambda x: A.HorizontalFlip() if x else None,
    "resize": lambda x: A.Resize(height=x[0], width=x[1]),
    "rotate": lambda x: A.Rotate(limit=x),
    "color_jitter": lambda x: A.ColorJitter(
        brightness=x[0], contrast=x[1], saturation=x[2], hue=x[3]
    ),
}



