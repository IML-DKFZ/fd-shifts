import torch
import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self, cf):
        super(ViT, self).__init__()

        self.encoder = Encoder(cf)
        self.classifier = self.encoder.model.head

    def forward(self, x):
        out = self.encoder(x)
        pred = self.classifier(out)
        return pred


class Encoder(nn.Module):
    def __init__(self, cf):
        super(Encoder, self).__init__()
        # name = cf.model.network.name if "vit" in cf.model.network.name else cf.model.network.backbone
        # print("Init VGG type:{}".format(name))
        num_classes = cf.data.num_classes
        if cf.eval.ext_confid_name == "dg":
            num_classes += 1

        self.model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=True,
            img_size=cf.data.img_size[0],
            num_classes=num_classes,
            drop_rate=cf.model.dropout_rate,
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
