import fd_shifts.models.networks as networks
from torch import nn
from torch.nn import functional as F


class DeVriesAndEncoder(nn.Module):
    def __init__(self, cf):
        super().__init__()

        network = networks.get_network(cf.model.network.backbone)(
            cf
        )  # todo make arguments explcit!
        self.encoder = network.encoder
        self.classifier = network.classifier
        self.devries_net = DeVriesNet(cf)  # todo make arguments explcit!

    def forward(self, x):

        x = self.encoder(x)
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
