import torch
import torch.nn as nn

class DecisionNetwork(nn.Module):
    def __init__(self):
        super(DecisionNetwork, self).__init__()
        self.layer1 = nn.Linear(32, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(p=0.25)
        self.layer3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.layer3(x)
        return x  