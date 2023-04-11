"""VGG11/13/16/19 in Pytorch."""
# modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch
import torch.nn as nn
from torch.autograd import Variable

from fd_shifts import configs, logger
from fd_shifts.models.networks.network import DropoutEnablerMixin, Network

cfg = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
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
    "vgg19": [
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


class VGG(Network):
    def __init__(self, cf: configs.Config):
        super(VGG, self).__init__()
        num_classes = cf.data.num_classes
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1
        self._encoder = Encoder(cf)
        self._classifier = Classifier(cf.model.fc_dim, num_classes)

    @property
    def encoder(self) -> DropoutEnablerMixin:
        return self._encoder

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    def forward(self, x):
        out = self.encoder(x)
        pred = self.classifier(out)
        return pred


class Encoder(DropoutEnablerMixin):
    def __init__(self, cf: configs.Config):
        super(Encoder, self).__init__()
        name = (
            cf.model.network.name
            if "vgg" in cf.model.network.name
            else cf.model.network.backbone
        )
        logger.info("Init VGG type:{}".format(name))
        self.dropout_rate = cf.model.dropout_rate
        self.fc_dim = cf.model.fc_dim
        self.avg_pool = cf.model.avg_pool
        self.features = self._make_layers(cfg[name])

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for ix, x in enumerate(cfg):
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]

                if self.dropout_rate > 0 and cfg[ix + 1] != "M":
                    rate = 0.3 if ix == 0 else 0.4
                    layers += [nn.Dropout(self.dropout_rate * rate)]
                in_channels = x
        if self.avg_pool:
            layers += [nn.AvgPool2d(kernel_size=1, stride=1), Flatten()]
        else:
            layers += [
                nn.Dropout(self.dropout_rate * 0.5),
                Flatten(),
                nn.Linear(512, self.fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate * 0.5),
            ]

        return nn.Sequential(*layers)

    def disable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], torch.nn.modules.dropout.Dropout):
                layer[1].eval()

    def enable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], torch.nn.modules.dropout.Dropout):
                layer[1].train()

    def forward(self, x):
        x = self.features(x)
        return x


class Classifier(nn.Module):
    def __init__(self, fc_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        return self.fc2(x)


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out
