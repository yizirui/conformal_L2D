import torch
import torch.nn as nn
import torch.nn.functional as F


class NetSimple(nn.Module):
    """
    Simple 2 layer CNN with fully connected relu layers
    NetSimple(n_dataset) instantiates one such model
    with paramters to the max, this can get close to 80% accuracy
    """

    def __init__(self, num_classes, width1=6, width2=16, ff_units1=120, ff_units2=84):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(width1, width2, 5)
        self.fc1 = nn.Linear(width2 * 5 * 5, ff_units1)
        self.fc2 = nn.Linear(ff_units1, ff_units2)
        self.fc3 = nn.Linear(ff_units2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
