import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .pool import *
from .modules.att import *

class Addition(nn.Module):
    def __init__(self):
        super(Addition, self).__init__()
    
    def forward(self, x1, x2):
        return x1 + x2

class PoolCALayer(nn.Module):
    def __init__(self, levels, mode='avg'):
        super(PoolCALayer, self).__init__()
        # global average pooling: feature --> point
        self.pool = SpatialPyramidPooling(levels, mode=mode)
        # feature channel downscale and upscale --> channel weight
        self.dot_att = ScaleDotProductAttention(self.pool.get_output_size()) 

    def forward(self, d, u):
        bs, num_down_filter, w, h = d.size()
        d_pool = self.pool(d) # [bs, num_down_filter, sum(level**2)]
        u_pool = self.pool(u) # [bs, num_up_filter, sum(level**2)]

        y = self.dot_att(u_pool, d_pool, d.view(bs, num_down_filter, -1))
        return y.view(bs, num_down_filter, w, h)



#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

