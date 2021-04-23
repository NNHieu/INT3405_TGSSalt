import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .modules.att import SpatialAttention2d, GAB

class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))
        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))
        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
                         nn.BatchNorm2d(output_dim),
                         nn.ELU(True))

class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)
        return x

class ResEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ResEncoder, self).__init__()
        self.resnet = torchvision.models.resnet34(True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # self.encode2 = nn.Sequential(self.resnet.layer1,
        #                             SCse(64))
        # self.encode3 = nn.Sequential(self.resnet.layer2,
        #                              SCse(128))
        # self.encode4 = nn.Sequential(self.resnet.layer3,
        #                              SCse(256))
        # self.encode5 = nn.Sequential(self.resnet.layer4,
        #                              SCse(512))
        self.encode2 = self.resnet.layer1
        self.encode3 = self.resnet.layer2
        self.encode4 = self.resnet.layer3
        self.encode5 = self.resnet.layer4


    def forward(self, x):
            # x: batch_size, 1, 25, 64
            x = self.conv1(x)       # 64, 25, 64
            e2 = self.encode2(x)    # 64, 25, 64
            e3 = self.encode3(e2)   # 128, 12, 32
            e4 = self.encode4(e3)   # 256, 6, 16
            e5 = self.encode5(e4)   # 512, 3, 8
            return e2, e3, e4, e5
            


class DecoderBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(DecoderBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        # self.s_att = SpatialAttention2d(n_out)
        # self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        
        return cat_p
        # s = self.s_att(cat_p)
        # c = self.c_att(cat_p)
        # return s + c

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        planes = [64, 128, 256, 512]
        self.center = nn.Sequential(FPAv2(planes[3], planes[2]),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = DecoderBlock(planes[2], planes[3], planes[0])
        self.decode4 = DecoderBlock(planes[0], planes[2], planes[0])
        self.decode3 = DecoderBlock(planes[0], planes[1], planes[0])
        self.decode2 = DecoderBlock(planes[0], planes[0], planes[0])

        self.logit = nn.Sequential(nn.Conv2d(planes[0] * 4, planes[0] // 2, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(planes[0] // 2, 1, kernel_size=1, bias=False))

    def forward(self, e2, e3, e4, e5):
        f = self.center(e5)  # 256, 8, 8

        d5 = self.decode5(f, e5)  # 64, 16, 16
        d4 = self.decode4(d5, e4)  # 64, 32, 32
        d3 = self.decode3(d4, e3)  # 64, 64, 64
        d2 = self.decode2(d3, e2)  # 64, 128, 128

        # f = d2
        f = torch.cat((d2,
                       F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True),
                       F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True),
                       F.interpolate(d5, scale_factor=8, mode='bilinear', align_corners=True)), 1)  # 256, 128, 128

        f = F.dropout2d(f, p=0.4)
        logit = self.logit(f)  # 1, 128, 128
        return logit

# stage3 model
class Res34Unet(nn.Module):
    def __init__(self):
        super(Res34Unet, self).__init__()
        self.encoder = ResEncoder(1)
        self.decoder = Decoder()
                    
    def forward(self, x):
        e2, e3, e4, e5 = self.encoder(x)
        logit = self.decoder(e2, e3, e4, e5)  # 1, 128, 128
        return logit