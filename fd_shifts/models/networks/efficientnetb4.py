import torch
import torch.nn as nn
import timm


class EfficientNetb4(nn.Module):
    def __init__(self, cf):
        super(EfficientNetb4, self).__init__()

        self.encoder = Encoder(cf)
        # self.classifier = Classifier(self.encoder.model.head)
        self.num_features = self.encoder.model.num_features

    def forward(self, x):
        # out = self.encoder(x)
        # pred = self.classifier(out)
        return self.encoder(x)

    def forward_features(self, x):
        x = self.encoder.model.forward_features(x)
        avg2d = torch.nn.AvgPool2d(16, stride=1)
        flatten = torch.nn.Flatten()
        return flatten(avg2d(x))

    def head(self, x):
        return self.encoder.model.classifier(x)


class Encoder(nn.Module):
    def __init__(self, cf):
        super(Encoder, self).__init__()
        # name = cf.model.network.name if "vit" in cf.model.network.name else cf.model.network.backbone
        # print("Init VGG type:{}".format(name))
        num_classes = cf.data.num_classes
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1

        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            # img_size=cf.data.img_size[0],
            num_classes=num_classes,
            drop_rate=cf.model.dropout_rate * 0.1,
        )
        self.model.reset_classifier(num_classes)
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
        x = self.model(x)
        return x

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
