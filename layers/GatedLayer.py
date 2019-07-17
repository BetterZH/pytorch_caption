import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GatedTanh(nn.Module):
    def __init__(self, size):
        super(GatedTanh, self).__init__()
        self.layer1 = nn.Linear(size, size)
        self.layer2 = nn.Linear(size, size)

    # input: x
    def forward(self, x):

        y_1 = F.tanh(self.layer1(x))
        g = F.sigmoid(self.layer2(x))
        y = y_1 * g

        return y