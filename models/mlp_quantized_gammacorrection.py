import torch
import torch.nn as nn
from .modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 8
BIPRECISION = True
__all__ = ['mlp_quantized_gammacorrection']

class twomlp_quantized_model(nn.Module):

    def __init__(self):
        super(twomlp_quantized_model, self).__init__()
        self.fc0 = QLinear(16, 16, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)

        self.fc1 = QLinear(16, 12, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.fc0(inputs)
        out = self.relu(out)
        out = self.fc1(out)
        return out


def mlp_quantized_gammacorrection(**kwargs):
    return twomlp_quantized_model()
