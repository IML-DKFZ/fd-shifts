from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, cf):
        super(MLP, self).__init__()

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

        self.fc1 = nn.Linear(
            self.img_size[0] * self.img_size[1],
            self.fc_dim,
        )
        self.fc_dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        if self.eval_mcdropout:
            x = F.dropout(x, 0.3, training=True)
        else:
            x = self.fc_dropout(x)

        return x


class Classifier(nn.Module):
    def __init__(self, cf):
        super(Classifier, self).__init__()

        self.num_classes = cf.data.num_classes
        self.fc_dim = cf.model.fc_dim
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x):
        return self.fc2(x)
