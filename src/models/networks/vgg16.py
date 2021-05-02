import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch


class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
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



class VGG16(nn.Module):
    def __init__(self, cf):
        super(VGG16, self).__init__()

        self.encoder = Encoder(cf)
        self.classifier = Classifier(cf)

    def forward(self, x):

        x = self.encoder(x)
        x = self.classifier(x)

        return x


class Encoder(nn.Module):
    def __init__(self, cf):
        super(Encoder, self).__init__()

        self.img_size = cf.data.img_size
        self.fc_dim = cf.model.fc_dim
        self.eval_mcdropout = False
        self.dropout_flag = cf.model.dropout_flag
        print("CHECK DROPOUT IN VGG", self.dropout_flag)

        self.conv1 = Conv2dSame(self.img_size[-1], 64, 3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_dropout = nn.Dropout(0.3)
        self.conv2 = Conv2dSame(64, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = Conv2dSame(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_dropout = nn.Dropout(0.4)
        self.conv4 = Conv2dSame(128, 128, 3)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = Conv2dSame(128, 256, 3)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv5_dropout = nn.Dropout(0.4)
        self.conv6 = Conv2dSame(256, 256, 3)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv6_dropout = nn.Dropout(0.4)
        self.conv7 = Conv2dSame(256, 256, 3)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv8 = Conv2dSame(256, 512, 3)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv8_dropout = nn.Dropout(0.4)
        self.conv9 = Conv2dSame(512, 512, 3)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv9_dropout = nn.Dropout(0.4)
        self.conv10 = Conv2dSame(512, 512, 3)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv11 = Conv2dSame(512, 512, 3)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv11_dropout = nn.Dropout(0.4)
        self.conv12 = Conv2dSame(512, 512, 3)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.conv12_dropout = nn.Dropout(0.4)
        self.conv13 = Conv2dSame(512, 512, 3)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(2)

        self.end_dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, self.fc_dim)
        self.dropout_fc = nn.Dropout(0.5)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.3, training=True)
        elif self.dropout_flag:
            out = self.conv1_dropout(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = self.maxpool1(out)

        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.4, training=True)
        elif self.dropout_flag:
            out = self.conv3_dropout(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = self.maxpool2(out)

        out = F.relu(self.conv5(out))
        out = self.conv5_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.4, training=True)
        elif self.dropout_flag:
            out = self.conv5_dropout(out)
        out = F.relu(self.conv6(out))
        out = self.conv6_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.4, training=True)
        elif self.dropout_flag:
            out = self.conv6_dropout(out)
        out = F.relu(self.conv7(out))
        out = self.conv7_bn(out)
        out = self.maxpool3(out)

        out = F.relu(self.conv8(out))
        out = self.conv8_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.4, training=True)
        elif self.dropout_flag:
            out = self.conv8_dropout(out)
        out = F.relu(self.conv9(out))
        out = self.conv9_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.4, training=True)
        elif self.dropout_flag:
            out = self.conv9_dropout(out)
        out = F.relu(self.conv10(out))
        out = self.conv10_bn(out)
        out = self.maxpool4(out)

        out = F.relu(self.conv11(out))
        out = self.conv11_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.4, training=True)
        elif self.dropout_flag:
            out = self.conv11_dropout(out)
        out = F.relu(self.conv12(out))
        out = self.conv12_bn(out)
        if self.eval_mcdropout:
            out = F.dropout(out, 0.4, training=True)
        elif self.dropout_flag:
            out = self.conv12_dropout(out)
        out = F.relu(self.conv13(out))
        out = self.conv13_bn(out)
        out = self.maxpool5(out)

        if self.eval_mcdropout:
            out = F.dropout(out, 0.5, training=True)
        elif self.dropout_flag:
            out = self.end_dropout(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out)) # todo average pool in devries?
        if self.eval_mcdropout:
            out = F.dropout(out, 0.5, training=True)
        elif self.dropout_flag:
            out = self.dropout_fc(out)

        return out

    def load_pretrained_imagenet_params(self, imagenet_weights_path):
        print("loading imagenet parameters into vgg16")
        pretrained_model = models.vgg16(pretrained=False).cuda()
        pretrained_model.load_state_dict(torch.load(imagenet_weights_path))

        pretrained_layers = []
        for _layer in pretrained_model.features.children():
            if isinstance(_layer, nn.Conv2d):
                pretrained_layers.append(_layer)

        model_layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7,
            self.conv8,
            self.conv9,
            self.conv10,
            self.conv11,
            self.conv12,
            self.conv13,
        ]

        assert len(pretrained_layers) == len(model_layers)

        for l1, l2 in zip(pretrained_layers, model_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

class Classifier(nn.Module):
    def __init__(self, cf):
        super(Classifier, self).__init__()

        self.num_classes = cf.data.num_classes
        self.fc_dim = cf.model.fc_dim
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x):
        return self.fc2(x)