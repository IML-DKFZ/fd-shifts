import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4

from fd_shifts.models.networks.network import DropoutEnablerMixin, Network


class EfficientNetb4(Network):
    def __init__(self, cf):
        super(EfficientNetb4, self).__init__()

        self._encoder = Encoder(cf)
        self._classifier = Classifier(self._encoder.model.classifier)
        # self.classifier = Classifier(self.encoder.model.head)
        # self.num_features = self.encoder.model.classifier[1].in_features

    def forward(self, x):
        out = self._encoder(x)
        pred = self._classifier(out)
        return pred

    # def head(self, x):
    #    return self.encoder.model.classifier(x)
    def forward_features(self, x):
        return self._encoder(x)

    def head(self, x):
        return self._classifier(x)


class Encoder(DropoutEnablerMixin):
    def __init__(self, cf):
        super(Encoder, self).__init__()
        # name = cf.model.network.name if "vit" in cf.model.network.name else cf.model.network.backbone
        # print("Init VGG type:{}".format(name))
        num_classes = cf.data.num_classes
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1
        self.model = efficientnet_b4(pretrained=True)
        self.dropout_rate = cf.model.dropout_rate * 0.2

        in_features = cf.model.fc_dim
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    def disable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].eval()

    def enable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].train()

    def forward(self, x):
        x = self.model.features(x)
        avg2d = torch.nn.AvgPool2d(16, stride=1)
        flatten = torch.nn.Flatten()
        return flatten(avg2d(x))

    # def forward(self, x):
    #    x = self.model(x)
    #    return x

    # def load_state_dict(self, state_dict, strict=True):
    #     print(state_dict)
    #     self.model.load_state_dict(state_dict, strict)


class Classifier(nn.Module):
    def __init__(self, module):
        super(Classifier, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)

    def load_state_dict(self, state_dict, strict=True):
        pass
