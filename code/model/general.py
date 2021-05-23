import torch.nn as nn
from .activation import Mish

def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
                         nn.BatchNorm2d(output_dim),
                         Mish())