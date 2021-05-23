import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
from .lovasz_losses import lovasz_hinge, lovasz_hinge2

def dice_loss(inputs, targets, smooth=1.):
    inputs = torch.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()                            
    loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    return loss

def dice_bce_loss(inputs, targets, smooth=1.):
    inputs = torch.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()                            
    loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
    return loss + bce


class FocalLoss(nn.modules.loss._Loss):
    def __init__(self, gamma=2., alpha=1., weight = None, size_average=None, reduce=None, reduction = 'mean',
                 pos_weight = None) -> None:
        super(FocalLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.use_pw = True
    
    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def forward(self, input, target):
        assert self.weight is None or isinstance(self.weight, Tensor)
        assert self.pos_weight is None or isinstance(self.pos_weight, Tensor)
        logpt = -F.binary_cross_entropy_with_logits(input, target,
#                                                       self.weight,
                                                      pos_weight=self.pos_weight if self.use_pw else None,
                                                      reduction='none')
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( self.alpha * (1-pt)**self.gamma ) * logpt
        if self.weight is not None:
            focal_loss *= self.weight
        return self.reduce_loss(focal_loss)

class LovaszHingeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, target):
        return lovasz_hinge(input, target)


def get_loss_func(name):
    if name == 'bce':
        return F.binary_cross_entropy_with_logits
    elif name == 'dice':
        return dice_loss
    elif name == 'dice_bce':
        return dice_bce_loss
    elif name == 'focal':
        return FocalLoss()
    elif name == 'lovasz_hinge':
        return FocalLoss()
    elif name == 'lovasz_hinge_now':
        return lovasz_hinge
    else:
        print('Error: ', name, ' is not defined.')
        return