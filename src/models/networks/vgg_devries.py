"""VGG11/13/16/19 in Pytorch."""
# modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch.nn as nn
import torch
from torch.autograd import Variable


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG13(nn.Module):
    def __init__(self, cf):
        super(VGG13, self).__init__()
        self.encoder = Encoder(cf)
        self.classifier = Classifier(cf)

    def forward(self, x):
        out = self.encoder(x)
        pred = self.classifier(out)
        return pred


class Encoder(nn.Module):
    def __init__(self, cf):
        super(Encoder, self).__init__()
        self.features = self._make_layers(cfg["VGG13"])

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(3).squeeze(2)
        return x


class Classifier(nn.Module):
    def __init__(self, cf):
        super(Classifier, self).__init__()

        self.num_classes = cf.data.num_classes
        self.fc_dim = cf.model.fc_dim
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x):
        return self.fc2(x)
