from __future__ import annotations

import timm
import torch
import torch.nn as nn

from fd_shifts import configs
from fd_shifts.models.networks.network import DropoutEnablerMixin, Network


class ViT(Network):
    def __init__(self, cf: configs.Config):
        super().__init__()

        self._encoder = Encoder(cf)
        self._classifier = Classifier(self.encoder.model.head)

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def classifier(self) -> Classifier:
        return self._classifier

    def forward(self, x):
        out = self.encoder(x)
        pred = self.classifier(out)
        return pred


class Encoder(DropoutEnablerMixin):
    def __init__(self, cf: configs.Config):
        super().__init__()
        num_classes = cf.data.num_classes
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1

        self.model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=True,
            img_size=cf.data.img_size[0],
            num_classes=num_classes,
            drop_rate=cf.model.dropout_rate * 0.1,
        )
        self.model.reset_classifier(num_classes)
        self.model.head.weight.tensor = torch.zeros_like(self.model.head.weight)
        self.model.head.bias.tensor = torch.zeros_like(self.model.head.bias)
        self.dropout_rate = cf.model.dropout_rate

    def disable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].eval()

    def enable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].train()

    def forward(self, x):
        x = self.model.forward_features(x)
        return x


class Classifier(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)

    def load_state_dict(self, state_dict, strict=True):
        pass
