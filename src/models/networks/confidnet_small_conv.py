
from src.models.networks.small_conv import Encoder
from src.models.networks.small_conv import Classifier

from torch import nn
from torch.nn import functional as F
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

class ConfidNetAndENcoder2(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.feature_dim = cf.model.fc_dim
        self.conv1 = Conv2dSame(3, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSame(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = Conv2dSame(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = Conv2dSame(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv5 = Conv2dSame(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = Conv2dSame(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(2048, self.feature_dim)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.feature_dim, 10)

        self.uncertainty1 = nn.Linear(self.feature_dim, 400)
        self.uncertainty2 = nn.Linear(400, 400)
        self.uncertainty3 = nn.Linear(400, 400)
        self.uncertainty4 = nn.Linear(400, 400)
        self.uncertainty5 = nn.Linear(400, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = self.maxpool1(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout1(out)

        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = self.maxpool2(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout2(out)

        out = F.relu(self.conv5(out))
        out = self.conv5_bn(out)
        out = F.relu(self.conv6(out))
        out = self.conv6_bn(out)
        out = self.maxpool3(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout3(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout4(out)

        uncertainty = F.relu(self.uncertainty1(out))
        uncertainty = F.relu(self.uncertainty2(uncertainty))
        uncertainty = F.relu(self.uncertainty3(uncertainty))
        uncertainty = F.relu(self.uncertainty4(uncertainty))
        uncertainty = self.uncertainty5(uncertainty)
        pred = self.fc2(out)
        return uncertainty

class ConfidNetAndENcoder(nn.Module):
    def __init__(self, cf):
        super().__init__()

        confid_net_fc_dim = 400 # 400
        self.encoder = Encoder(cf)
        # self.confid_net = ConfidNet(cf, confid_net_fc_dim)

        self.uncertainty1 = nn.Linear(cf.model.fc_dim, confid_net_fc_dim)
        self.uncertainty2 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty3 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty4 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty5 = nn.Linear(cf.model.fc_dim, 1)
        self.reps = torch.ones(128, cf.model.fc_dim).cuda()

    def forward(self, x):

        reps = self.encoder(x)
        # # confid = self.confid_net(reps)
        # x = F.relu(self.uncertainty1(reps))
        # x = F.relu(self.uncertainty2(x))
        # x = F.relu(self.uncertainty3(x))
        # x = F.relu(self.uncertainty4(x))
        # reps = torch.ones(c)
        confid = self.uncertainty5(reps)
        # confid = x
        # confid = F.sigmoid(confid)
        print(confid.size(), "SIZE")
        return confid


class ConfidNet(nn.Module):
    def __init__(self, cf, confid_net_fc_dim):
        super().__init__()


        self.uncertainty1 = nn.Linear(cf.model.fc_dim, confid_net_fc_dim)
        self.uncertainty2 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty3 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty4 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty5 = nn.Linear(confid_net_fc_dim, 1)

    def forward(self, x):

        # x = F.relu(self.uncertainty1(x))
        # x = F.relu(self.uncertainty2(x))
        # x = F.relu(self.uncertainty3(x))
        # x = F.relu(self.uncertainty4(x))
        confid = self.uncertainty5(x)

        return confid