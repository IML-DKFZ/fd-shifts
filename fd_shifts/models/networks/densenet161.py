from torchvision import models
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, List, Tuple
import torch.nn.functional as F
import torchvision


class Densenet161(nn.Module):
    def __init__(self, cf) -> None:
        super(Densenet161, self).__init__()
        self.model = models.densenet161(pretrained=True)
        self.dropout_rate = cf.model.dropout_rate * 0.1
        for layer in self.named_modules():
            if isinstance(layer[1], torchvision.models.densenet._DenseLayer):
                layer[1].drop_rate = self.dropout_rate
        self.model.features[0] = nn.Conv2d(
            in_channels=6,
            out_channels=96,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.encoder = Encoder(model=self.model)
        self.classifier = Classifier(cf=cf, model=self.model)

    def forward(self, x):
        out = self.encoder(x)
        pred = self.classifier(out)
        return pred

    # def head(self, x):
    #    return self.encoder.model.classifier(x)
    def forward_features(self, x):
        return self.encoder.forward(x)

    def head(self, x):
        return self.classifier.forward(x)


class Encoder(nn.Module):
    def __init__(self, model) -> None:
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def disable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], torchvision.models.densenet._DenseLayer):
                layer[1].drop_rate = 0

    def enable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], torchvision.models.densenet._DenseLayer):
                layer[1].drop_rate = 0.1


class Classifier(nn.Module):
    def __init__(self, cf, model) -> None:
        super(Classifier, self).__init__()
        num_ftrs = model.classifier.in_features
        num_classes = cf.data.num_classes
        self.model = model
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1
        self.model.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.model.classifier(x)
