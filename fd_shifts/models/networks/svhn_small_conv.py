from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fd_shifts import configs
from fd_shifts.models.networks.network import DropoutEnablerMixin, Network


class Conv2dSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        padding_layer: nn.Module = nn.ReflectionPad2d,
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class SmallConv(Network):
    def __init__(self, cf: configs.Config):
        super().__init__()
        num_classes = cf.data.num_classes
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1

        self._encoder = Encoder(cf)
        self._classifier = Classifier(cf.model.fc_dim, num_classes)

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def classifier(self) -> Classifier:
        return self._classifier

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)

        return x


class Encoder(DropoutEnablerMixin):
    def __init__(self, cf: configs.Config):
        super().__init__()

        self.img_size = cf.data.img_size
        self.fc_dim = cf.model.fc_dim
        self.dropout_rate = cf.model.dropout_rate * 0.3
        self.eval_mcdropout = False

        self.conv1 = Conv2dSame(self.img_size[-1], 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSame(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.conv3 = Conv2dSame(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = Conv2dSame(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.conv5 = Conv2dSame(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = Conv2dSame(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.fc1 = nn.Linear(2048, self.fc_dim)
        self.dropout4 = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.maxpool1(x)
        if self.dropout_rate > 0:
            x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = F.relu(self.conv4(x))
        x = self.conv4_bn(x)
        x = self.maxpool2(x)
        if self.dropout_rate > 0:
            x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.conv5_bn(x)
        x = F.relu(self.conv6(x))
        x = self.conv6_bn(x)
        x = self.maxpool3(x)
        if self.dropout_rate > 0:
            x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.dropout_rate > 0:
            x = self.dropout4(x)

        return x

    def disable_dropout(self) -> None:
        for layer in self.named_modules():
            if isinstance(layer[1], torch.nn.modules.dropout.Dropout):
                layer[1].eval()

    def enable_dropout(self) -> None:
        for layer in self.named_modules():
            if isinstance(layer[1], torch.nn.modules.dropout.Dropout):
                layer[1].train()


class Classifier(nn.Module):
    def __init__(self, fc_dim: int, num_classes: int):
        super().__init__()

        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        return self.fc2(x)
