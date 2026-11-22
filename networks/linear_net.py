import torch.nn as nn


class LinearNet(nn.Module):
    """
    Linear Classifier with out units and no softmax
    """

    def __init__(self, input_dim, out_dim):
        super(LinearNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        out = self.fc(x)
        return out
