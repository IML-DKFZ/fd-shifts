import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["VGG", "vgg16", "vgg16_bn"]


class VGG(nn.Module):
    def __init__(self, cf):
        super(VGG, self).__init__()

        num_classes = cf.data.num_classes
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1
        input_size = cf.data.img_size[0]
        dropout_rate = cf.model.dropout_rate
        self.encoder = make_layers(cfg["D"], batch_norm=True, dropout_rate=dropout_rate)

        if input_size == 32:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout2d(0.5 * dropout_rate),
                nn.Linear(512, num_classes),
            )
        elif input_size == 64:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout2d(0.5 * dropout_rate),
                nn.Linear(512, num_classes),
            )
        self._initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, dropout_rate=1):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif type(v) == int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v) == float:
            layers += [nn.Dropout2d(v * dropout_rate)]
    return nn.Sequential(*layers)


cfg = {
    "D": [
        64,
        0.3,
        64,
        "M",
        128,
        0.4,
        128,
        "M",
        256,
        0.4,
        256,
        0.4,
        256,
        "M",
        512,
        0.4,
        512,
        0.4,
        512,
        "M",
        512,
        0.4,
        512,
        0.4,
        512,
        "M",
        0.5,
    ]
}
