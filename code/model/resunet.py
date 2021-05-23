import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .general import *

class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        self.resnet = torchvision.models.resnet34(True)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)
        # self.conv1[0].stride = (1, 1)
        self.encode2 = self.resnet.layer1
        self.encode3 = self.resnet.layer2
        self.encode4 = self.resnet.layer3
        self.encode5 = self.resnet.layer4
        mean = list(reversed([0.485, 0.456, 0.406]))
        std = list(reversed([0.229, 0.224, 0.225]))
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))


    def forward(self, x):
            # x: batch_size, 1, 25, 64
            # IMAGENET_MEAN = (0.485, 0.456, 0.406)
            # IMAGENET_STD = (0.229, 0.224, 0.225)
            x = x.repeat(1, 3, 1, 1)
            x = (x - self.mean) / self.std
            # x = torch.cat([ (x - 0.406)/0.225,
            #                 (x - 0.456)/0.224,
            #                 (x - 0.485)/0.229,                
            #               ], 1)
            x = self.conv1(x)       
            e2 = self.encode2(x)    
            e3 = self.encode3(e2)   
            e4 = self.encode4(e3)   
            e5 = self.encode5(e4)   
            return e2, e3, e4, e5

    @classmethod
    def planes(cls):
        return  [64, 128, 256, 512]

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        # print(x.device)
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
        self.dblocks = nn.Sequential(*DecoderBlock.build(planes))
        self.last_dblock = DecoderBlock(planes[0], planes[0] // 2, planes[0])

    def forward(self, center, *enc_feature_block):
        decode_feature_block = []
        f = center
        for i, dblock in enumerate(self.dblocks):
            f = dblock(f, enc_feature_block[i])
            decode_feature_block.append(f)
        decode_feature_block.append(self.last_dblock(f))
        return decode_feature_block

# stage3 model
class Res34Unet(nn.Module):
    def __init__(self):
        super(Res34Unet, self).__init__()
        self.encoder = ResEncoder()
        planes = ResEncoder.planes()
        self.center = nn.Sequential(
            conv3x3(planes[-1], planes[-1]),
            conv3x3(planes[-1], planes[-2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = Decoder(planes)
        self.logit = nn.Sequential(nn.Conv2d(planes[0] * 5, planes[0], kernel_size=3, padding=1),
                                    Mish(),
                                    nn.Conv2d(planes[0], 1, kernel_size=1, bias=False))

    def forward(self, x):
        e2, e3, e4, e5 = self.encoder(x)
        c = self.center(e5)
        d5, d4, d3, d2, d1 = self.decoder(c, e5, e4, e3, e2)
        f = torch.cat((d1,
                       F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 128, 128

        f = F.dropout2d(f, p=0.4)
        logit = self.logit(f)  # 1, 128, 128
        return logit