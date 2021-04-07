
from src.models.networks.small_conv import Encoder
from src.models.networks.small_conv import Classifier
from torch import nn
from torch.nn import functional as F


class ConfidNetAndENcoder(nn.Module):
    def __init__(self, cf):
        super().__init__()

        self.encoder = Encoder(cf) # todo make arguments explcit!
        self.confid_net = ConfidNet(cf) # todo make arguments explcit!
        self.classifier = Classifier(cf)

    def forward(self, x):

        x = self.encoder(x)
        pred_class = self.classifier(x)
        pred_confid = self.confid_net(x)

        return pred_class, pred_confid


class ConfidNet(nn.Module):
    def __init__(self, cf):
        super().__init__()

        confid_net_fc_dim = cf.model.confidnet_fc_dim
        self.uncertainty1 = nn.Linear(cf.model.fc_dim, confid_net_fc_dim)
        self.uncertainty2 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty3 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty4 = nn.Linear(confid_net_fc_dim, confid_net_fc_dim)
        self.uncertainty5 = nn.Linear(confid_net_fc_dim, 1)

    def forward(self, x):

        x = F.relu(self.uncertainty1(x))
        x = F.relu(self.uncertainty2(x))
        x = F.relu(self.uncertainty3(x))
        x = F.relu(self.uncertainty4(x))
        confid = self.uncertainty5(x)

        return confid