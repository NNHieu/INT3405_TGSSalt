import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from utils import calculate_mAP, str2bool
from model import *

class LitUNet(pl.LightningModule):
    def __init__(self, bilinear, **kwargs):
        super().__init__()
        self.save_hyperparameters()
#         self.model = UNet(1, 1, bilinear=bilinear)
        self.model = AttUNet(1, 1, bilinear=bilinear)
        
    def forward(self, images):
        return self.model(images)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        pred_mask = self.model(images)
        bs, w, h, n_class = pred_mask.size()
        loss = F.binary_cross_entropy_with_logits(pred_mask.view(bs, -1, n_class), masks.view(bs, -1, n_class))

        # Logs
        mAP = calculate_mAP(masks.detach(), pred_mask.detach())
        self.log('loss_total', {'train': loss.item()}, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('mAP', {'train': mAP.item()}, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # self.log('train/conf_loss', conf_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        pred_mask = self.model(images)
        bs, w, h, n_class = pred_mask.size()
        loss = F.binary_cross_entropy_with_logits(pred_mask.view(bs, -1, n_class), masks.view(bs, -1, n_class))

        # Logs
        mAP = calculate_mAP(masks.detach(), pred_mask.detach())
        self.log('loss_total', {'val': loss.item()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mAP', {'val': mAP.item()}, on_epoch=True, prog_bar=False, logger=True)
        self.log('metrics_mAP', mAP.item(), on_epoch=True, prog_bar=True, logger=False)

 
    def configure_optimizers(self):
        groups= self.create_param_groups()  
        optimizer = torch.optim.AdamW(params=groups,
                                      lr=self.hparams.lr, 
                                      weight_decay=self.hparams.weight_decay)
#         optimizer = torch.optim.SSD(params=params,
#                                         lr=self.hparams.head_base_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        
#         head_base_lrs = [2 * self.hparams.head_base_lr, self.hparams.head_base_lr]
#         head_max_lrs = [2 * self.hparams.head_max_lr, self.hparams.head_max_lr]
    
        return optimizer
    
    def create_param_groups(self):
        groups = [{'params': self.model.parameters()}]
        return groups
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False) 
        parser.add_argument("--bilinear", type=str2bool, 
                            help="Use bilinear for up sampling.")
        parser.add_argument('--lr', type=float)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        return parser