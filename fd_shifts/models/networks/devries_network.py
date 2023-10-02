from torch import nn
from torch.nn import functional as F

import fd_shifts.models.networks as networks


class DeVriesAndEncoder(networks.network.Network):
    def __init__(self, cf):
        super().__init__()

        network = networks.get_network(cf.model.network.backbone)(cf)
        self._encoder = network.encoder
        self._classifier = network.classifier
        self.devries_net = DeVriesNet(cf)

    @property
    def encoder(self) -> networks.network.DropoutEnablerMixin:
        return self._encoder

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    def forward(self, x):
        x = self.encoder(x)
        pred_class = self.classifier(x)
        pred_confid = self.devries_net(x)

        return pred_class, pred_confid

    def forward_features(self, x):
        return self.encoder(x)

    def head(self, x):
        pred_class = self.classifier(x)
        pred_confid = self.devries_net(x)

        return pred_class, pred_confid


class DeVriesNet(nn.Module):
    def __init__(self, cf):
        super().__init__()

        self.uncertainty1 = nn.Linear(cf.model.fc_dim, 1)

    def forward(self, x):
        confid = self.uncertainty1(x)

        return confid
