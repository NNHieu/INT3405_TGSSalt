import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .modules.att import FPA, SpatialAttention2d, GAB
import timm
from .general import conv3x3
from .activation import Mish


class EffEncoder(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super(EffEncoder, self).__init__()
        self.effnet = timm.create_model(model_name, features_only=True, pretrained=pretrained)
        self.effnet.conv_stem.stride = (1, 1)
        mean = list(reversed([0.485, 0.456, 0.406]))
        std = list(reversed([0.229, 0.224, 0.225]))
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.effnet(x)

class AttDecoderBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels, up_channels=None):
        super(AttDecoderBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        if up_channels is not None:
            self.spacial_attn = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.Sigmoid()
            )
            # self.relu = nn.ReLU(inplace=True)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            e = e * self.spacial_attn(x)
            x = torch.cat([x, e], 1)
        # x = self.gab(x)
        # print(x.device)
        x = F.dropout2d(x, p=0.1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    @classmethod
    def build(cls, planes):
        return [cls(planes[2] + planes[3], planes[3], planes[0], planes[2]),
                cls(planes[0] + planes[2], planes[2], planes[0], planes[0]),
                cls(planes[0] + planes[1], planes[1], planes[0], planes[0]),
                cls(planes[0] + planes[0], planes[0], planes[0], planes[0]),]

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        # x = self.gab(x)
        # print(x.device)
        x = F.dropout2d(x, p=0.1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    @classmethod
    def build(cls, planes):
        return [cls(planes[2] + planes[3], planes[3], planes[0]),
                cls(planes[0] + planes[2], planes[2], planes[0]),
                cls(planes[0] + planes[1], planes[1], planes[0]),
                cls(planes[0] + planes[0], planes[0], planes[0]),]


class Decoder(nn.Module):
    def __init__(self, planes):
        super(Decoder, self).__init__()
        self.dblocks = nn.Sequential(*AttDecoderBlock.build(planes))
        self.last_dblock = DecoderBlock(planes[0], planes[0] // 2, planes[0])

    def forward(self, center, *enc_feature_block):
        decode_feature_block = []
        f = center
        for i, dblock in enumerate(self.dblocks):
            f = dblock(f, enc_feature_block[i])
            decode_feature_block.append(f)
        decode_feature_block.append(self.last_dblock(f))
        return decode_feature_block


class EffUnet(nn.Module):
    def __init__(self, model_name, pretrain):
        super(EffUnet, self).__init__()
        self.encoder = EffEncoder(model_name, pretrain)
        planes = self.encoder.effnet.feature_info.channels()[1:]
        self.center = nn.Sequential(
            # conv3x3(planes[-1], planes[-1]),
            # conv3x3(planes[-1], planes[-2]),
            FPA(planes[-1], planes[-2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = Decoder(planes)
        self.logit = nn.Sequential(nn.Conv2d(planes[0] * 5, planes[0], kernel_size=3, padding=1),
                                    Mish(),
                                    nn.Conv2d(planes[0], 1, kernel_size=1, bias=False))

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        c = self.center(e5)
        d5, d4, d3, d2, d1 = self.decoder(c, e5, e4, e3, e2)
        f = torch.cat((d1,
                       F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 128, 128
        f = F.dropout2d(f, p=0.2)
        logit = self.logit(f)  # 1, 128, 128
        return logit