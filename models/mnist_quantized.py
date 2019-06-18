import torch
import torch.nn as nn
import math
from .modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN

__all__ = ['mnist_quantized']

NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 8
BIPRECISION = True
class mnist_model(nn.Module):

    def __init__(self):
        super(mnist_model, self).__init__()
        self.feats = nn.Sequential(
#            nn.Conv2d(1, 32, 5, 1, 1),
            QConv2d(1, 32, kernel_size=5, stride=1, padding=1,
                                 bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

#            nn.Conv2d(32, 64, 3,  1, 1),
            QConv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                 bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),

            nn.ReLU(True),
#            nn.Conv2d(64, 64, 3,  1, 1),
            QConv2d(64, 64, kernel_size=3, stride=1, padding=1,
                                 bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),

            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
#            nn.Conv2d(64, 128, 3, 1, 1),
            QConv2d(64, 128, kernel_size=3, stride=1, padding=1,
                                 bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),

            nn.ReLU(True)
        )

#        self.classifier = nn.Conv2d(128, 10, 1)
        self.classifier = QConv2d(128, 10, kernel_size=1, stride=1, padding=1,
                             bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)

        self.avgpool = nn.AvgPool2d(6, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 10)
        return out


def mnist_quantized(**kwargs):
    return mnist_model()
