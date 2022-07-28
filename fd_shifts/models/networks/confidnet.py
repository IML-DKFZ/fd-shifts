from torch import nn
from torch.nn import functional as F

import fd_shifts.models.networks as networks


class ConfidNetAndEncoder(networks.network.Network):
    def __init__(self, cf):
        super().__init__()

        network = networks.get_network(cf.model.network.backbone)(cf)
        self._encoder = network.encoder
        self._classifier = network.classifier
        self.confid_net = ConfidNet(cf)

    @property
    def encoder(self) -> networks.network.DropoutEnablerMixin:
        return self._encoder

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    def forward(self, x):
        x = self.encoder(x)
        pred_class = self.classifier(x)
        pred_confid = self.confid_net(x)

        return pred_class, pred_confid

    def forward_features(self, x):
        return self.encoder(x)


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
