import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class  MLP(nn.Module):
    def __init__(self, amount_of_features, amount_of_classes, hidden_size, amount_layers, dropout):
        super(MLP, self).__init__()
        layers = [
            ('Linear1',  nn.Linear(amount_of_features, hidden_size).cuda()),
            ('Relu1', nn.ReLU()),
            ('Dropout1', nn.Dropout(dropout))
        ]
        for i in range(2, amount_layers):
            layers.append(('Linear{0}'.format(i), nn.Linear(hidden_size, hidden_size).cuda()))
            layers.append(('Relu{0}'.format(i), nn.ReLU().cuda()))
            layers.append(('Dropout{0}'.format(i), nn.Dropout(dropout)))

        layers.append(('Linear{0}'.format(amount_layers), nn.Linear(hidden_size, amount_of_classes).cuda()))
        layers.append(('LogSoftMax', nn.LogSoftmax(dim=-1).cuda()))
        self.model = nn.Sequential(OrderedDict(layers)).cuda()

    def forward(self, x):
        return self.model(x)
