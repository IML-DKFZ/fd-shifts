from torchvision import models
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, List, Tuple
import torch.nn.functional as F


class Densenet121(nn.Module):
    def __init__(self, cf) -> None:
        super(Densenet121).__init__()
        self.model = models.densenet121(pretrained=True)

        self.encoder = Encoder(self.model)
        self.classifier = Classifier(cf, self.encoder)

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
        super(Encoder).__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)


class Classifier(nn.Module):
    def __init__(self, cf, model) -> None:
        super(Classifier).__init__()
        num_ftrs = model.classifier.in_features
        num_classes = cf.data.num_classes
        self.model = model
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        self.model.classifier(x)
