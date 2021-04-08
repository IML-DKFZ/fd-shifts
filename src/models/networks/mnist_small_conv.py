
from torch import nn
from torch.nn import functional as F


class SmallConv(nn.Module):
    def __init__(self, cf):
        super(SmallConv, self).__init__()

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

        self.conv1 = nn.Conv2d(self.img_size[-1], 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, self.fc_dim) # 128
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        if self.eval_mcdropout:
            x = F.dropout(x, 0.25, training=True)
        else:
            x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.eval_mcdropout:
            x = F.dropout(x, 0.5, training=True)
        else:
            x = self.dropout2(x)

        return x


class Classifier(nn.Module):
    def __init__(self, cf):
        super(Classifier, self).__init__()

        self.num_classes = cf.data.num_classes
        self.fc_dim = cf.model.fc_dim
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x):
        return self.fc2(x)