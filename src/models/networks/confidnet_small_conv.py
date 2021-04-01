
from src.models.networks.small_conv import Encoder
from src.models.networks.small_conv import Classifier

from torch import nn
from torch.nn import functional as F


class ConfidNetAndENcoder(nn.Module):
    def __init__(self, cf):
        super().__init__()

        self.encoder = Encoder(cf)
        self.confid_net = ConfidNet(cf)

    def forward(self, x):

        reps = self.encoder(x)
        confid = self.confid_net(reps)
        return confid


class ConfidNet(nn.Module):
    def __init__(self, cf):
        super().__init__()


        self.uncertainty1 = nn.Linear(cf.model.fc_dim, 400)
        self.uncertainty2 = nn.Linear(400, 400)
        self.uncertainty3 = nn.Linear(400, 400)
        self.uncertainty4 = nn.Linear(400, 400)
        self.uncertainty5 = nn.Linear(400, 1)

    def forward(self, x):

        x = F.relu(self.uncertainty1(x))
        x = F.relu(self.uncertainty2(x))
        x = F.relu(self.uncertainty3(x))
        x = F.relu(self.uncertainty4(x))
        confid = self.uncertainty5(x)

        return confid