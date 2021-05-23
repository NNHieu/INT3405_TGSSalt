import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation import Mish

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation='relu'):
        super().__init__()
        
        if activation == 'relu':
            act = nn.ReLU(inplace=False)
        elif activation == 'mish':
            act = Mish()

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            act,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
# class RDown(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             ResidualBlock(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, concat=True, activation='relu'):
        super().__init__()
        self.concat = concat
        factor = 1 if self.concat else 2
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels // factor, out_channels, in_channels // 2, activation=activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // factor, out_channels, activation=activation)


    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.concat:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)
    
class RUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResidualBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualBlock(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# class AttUp(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
        
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#         self.att = Attention_block(F_g=in_channels // 2, F_l=in_channels // 2,F_int=in_channels // 4)




#     def forward(self, x1, x2, out_att=False):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x2 = self.att(x1, x2, out_att)
#         if out_att:
#             x2, att_out = x2
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         if out_att:
#             return x, att_out
#         return x
    
# class CAttUp(nn.Module):
#     """Upscaling then double conv + channel attention"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
        
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#         self.catt = CALayer(in_channels, reduction=4)


#     def forward(self, x1, x2, out_att=False):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
        
#         x = torch.cat([x2, x1], dim=1)
#         x = self.catt(x)
#         x = self.conv(x)
#         return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

""" Full assembly of the parts to form the complete network """

# class FCN(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True, num_filter=8):
#         super(FCN, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.inc = DoubleConv(n_channels, num_filter)
#         self.down1 = Down(num_filter, num_filter * 2)
#         self.down2 = Down(num_filter * 2, num_filter * 4)
#         self.down3 = Down(num_filter * 4, num_filter * 8)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(num_filter * 8, num_filter * 16 // factor)
#         self.up1 = Up(num_filter * 16, num_filter * 8 // factor, bilinear, False)
#         self.up2 = Up(num_filter * 8, num_filter * 4 // factor, bilinear, False)
#         self.up3 = Up(num_filter * 4, num_filter * 2 // factor, bilinear, False)
#         self.up4 = Up(num_filter * 2, num_filter, bilinear, False)
#         self.outc = OutConv(num_filter, n_classes)

#     def forward(self, x):
#         x = self.inc(x)
#         x = self.down1(x)
#         x = self.down2(x)
#         x = self.down3(x)
#         x = self.down4(x)
#         x = self.up1(x)
#         x = self.up2(x)
#         x = self.up3(x)
#         x = self.up4(x)
#         logits = self.outc(x)
#         return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_filter1, num_down_stage, bilinear=True, activation='relu'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, num_filter1, activation=activation)
        self.downs = [Down(num_filter1 * 2**i, num_filter1 * 2**(i + 1), activation=activation) for i in range(num_down_stage - 1)]
        factor = 2 if bilinear else 1 # do bilinear khong half num feature
        self.downs.append(Down(num_filter1 * 2**(num_down_stage - 1), num_filter1 * 2**num_down_stage // factor, activation=activation))
        self.downs = nn.Sequential(*self.downs)

        self.ups = []
        # self.mid_scale_conv = []
        for i in range(num_down_stage - 1):
            self.ups.append(Up(num_filter1 * 2**(num_down_stage - i) , num_filter1 * 2**(num_down_stage -1 - i) // factor, bilinear, activation=activation))
            # self.mid_scale_conv.append(nn.Conv2d(num_filter1 * 2**(num_down_stage -1 - i) // factor, n_classes, 1))
        self.ups.append(Up(num_filter1 * 2 , num_filter1 , bilinear, activation=activation))
        self.ups = nn.Sequential(*self.ups)
        # self.mid_scale_conv = nn.Sequential(*self.mid_scale_conv)
        self.outc = OutConv(num_filter1, n_classes)

    def forward(self, x):
        # Contrating path
        feature_blocks = [self.inc(x)]
        for down_module in self.downs:
            feature_blocks.append(down_module(feature_blocks[-1]))

        # Upsampling path
        resample_block = feature_blocks[-1]
        # mid_out = []
        for i, up_module in enumerate(self.ups):
            resample_block = up_module(resample_block, feature_blocks[-i-2])
            # if i < len(self.ups) - 1:
            #     mid_out.append(self.mid_scale_conv[i](resample_block))
        # Segment
        logits = self.outc(resample_block)
        return logits



# class RUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(RUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         num_filter = 32
#         self.inc = DoubleConv(n_channels, num_filter)
#         self.down1 = RDown(num_filter, num_filter * 2)
#         self.down2 = RDown(num_filter * 2, num_filter * 4)
#         self.down3 = RDown(num_filter * 4, num_filter * 8)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(num_filter * 8, num_filter * 16 // factor)
        
#         self.up1 = RUp(num_filter * 16, num_filter * 8 // factor, bilinear)
#         self.up2 = RUp(num_filter * 8, num_filter * 4 // factor, bilinear)
#         self.up3 = RUp(num_filter * 4, num_filter * 2 // factor, bilinear)
#         self.up4 = RUp(num_filter * 2, num_filter, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

# class AttUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(AttUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
        
#         self.up1 = AttUp(1024, 512 // factor, bilinear)        
#         self.up2 = AttUp(512, 256 // factor, bilinear)
#         self.up3 = AttUp(256, 128 // factor, bilinear)
#         self.up4 = AttUp(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
    
#     def collect_att_out(self, out_att, collection, up_out):
#         if out_att:
#             x, att_out = up_out
#             collection.append(att_out)
#             return x
#         return up_out
        
#     def forward(self, x, out_att=False):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         att_outs = []
#         x = self.collect_att_out(out_att, att_outs, self.up1(x5, x4, out_att))
#         x = self.collect_att_out(out_att, att_outs, self.up2(x, x3, out_att))
#         x = self.collect_att_out(out_att, att_outs, self.up3(x, x2, out_att))
#         x = self.collect_att_out(out_att, att_outs, self.up4(x, x1, out_att))
#         logits = self.outc(x)
#         if out_att:
#             return logits, att_outs
#         return logits
    

# class GAttUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, att_class, bilinear=True):
#         super(GAttUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
        
#         self.up1 = att_class(1024, 512 // factor, bilinear)        
#         self.up2 = att_class(512, 256 // factor, bilinear)
#         self.up3 = att_class(256, 128 // factor, bilinear)
#         self.up4 = att_class(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
    
#     def collect_att_out(self, out_att, collection, up_out):
#         if out_att:
#             x, att_out = up_out
#             collection.append(att_out)
#             return x
#         return up_out
        
#     def forward(self, x, out_att=False):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         att_outs = []
#         x = self.collect_att_out(out_att, att_outs, self.up1(x5, x4, out_att))
#         x = self.collect_att_out(out_att, att_outs, self.up2(x, x3, out_att))
#         x = self.collect_att_out(out_att, att_outs, self.up3(x, x2, out_att))
#         x = self.collect_att_out(out_att, att_outs, self.up4(x, x1, out_att))
#         logits = self.outc(x)
#         if out_att:
#             return logits, att_outs
#         return logits