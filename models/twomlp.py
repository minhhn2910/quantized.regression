import torch
import torch.nn as nn
__all__ = ['twomlp']

class twomlp_model(nn.Module):

    def __init__(self):
        super(twomlp_model, self).__init__()
        self.fc0 = nn.Linear(16,16)
        self.fc1 = nn.Linear(16,16)
        self.fc2 = nn.Linear(16,4)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.fc0(inputs)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def twomlp(**kwargs):
    return twomlp_model()
