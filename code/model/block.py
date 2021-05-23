import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model.activation import Mish
from .pool import *
# from .att import *
from .layer import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x1 = self.double_conv(x)
        return self.out_conv(x1 + x)
        
    
class Attention_block(nn.Module):
    """
    Scale down feature map by attention with upsample feature map
    """
    
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x, att_out=False):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        if att_out:
            return x*psi, psi
        return x*psi


## Channel Attention (CA) Layer
## From https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PoolCAttUp(nn.Module):
    """Upscaling then double conv + channel attention"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.add_norm = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(in_channels // 2)
            )
            self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 2)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2),
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
            )
            self.add_norm = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(in_channels // 2)
            )
            self.add_norm_1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(in_channels // 2)
            )
            self.conv = DoubleConv(in_channels // 2, out_channels)
        self.catt = PoolCALayer([1, 3, 5, 7])


    def forward(self, x1, x2, out_att=False):
        x1 = self.up(x1)
        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        
        y = self.catt(x2, x1)
        # x = torch.cat([y, x1], dim=1)
        y = x1 + y
        o1 = self.add_norm(y)   
        o2 = self.conv(o1)
        if not self.bilinear:
            o2 = self.add_norm_1(o1 + o2)
        return o2
    