
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import cv2

from data.datamodule import SaltDM
from utils.metrics import cal_mAP, cal_mIoU
from model import get_model
from model.lovasz_losses import lovasz_hinge, lovasz_hinge2
from model.layer import DiceBCELoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging

from utils import str2bool

# Loss
from model.loss import get_loss_func, LovaszHingeLoss

# from torchcontrib.optim import SWA

class LitResUnet(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.salt = get_model(self.hparams.model)
        self.criterion = get_loss_func(self.hparams.loss_func)

    def forward(self, x):
        return self.salt(x)
    
    def _step_with_loss(self, batch, batch_idx):
        inputs, masks = batch
        logit = self(inputs)
        # bs = masks.size(0)
        # loss = self.criterion(logit.view(bs, -1, 1), masks.view(bs, -1, 1))
        loss = self.criterion(logit.squeeze(1), masks.squeeze(1))
        # for i, mo in enumerate(mid_outs):
        #     loss += (0.5/len(mid_outs)) * self.criterion(mo.squeeze(1), resize(masks, mo.size()[-2:]).squeeze(1))
        return loss, logit

    def training_step(self, batch, batch_idx):
        loss, logit = self._step_with_loss(batch, batch_idx)
        self.log('Loss/train', loss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        masks = batch[1]
        preds = torch.sigmoid(logit).detach()
        precision = cal_mAP(preds, masks.detach(), 0.5)
        self.log('Metrics_mAP/train', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logit = self._step_with_loss(batch, batch_idx)
        masks = batch[1]
        preds = torch.sigmoid(logit).detach()
        precision = cal_mAP(preds, masks.detach(), 0.5)
        # precision = precision.mean()
        self.log('Loss/val', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metrics_mAP/val', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # checkpoint callback
        self.log('precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        images = batch
        logit_null = self(images)
        preds_null = torch.sigmoid(logit_null).squeeze(1).detach()
        logit_flip = self(images.flip(-1))
        preds_flip = torch.sigmoid(logit_flip).squeeze(1).flip(-1).detach()
        return (preds_flip + preds_null) / 2
    
    def test_epoch_end(self, outputs):
        preds = torch.cat(outputs, dim=0).cpu().numpy()

        preds_101 = np.zeros((preds.shape[0], 101, 101), dtype=np.float32)
        for idx in range(preds.shape[0]):
            preds_101[idx] = cv2.resize(preds[idx], dsize=(101, 101))
        np.save(self.hparams.save_pred, preds_101)

    def configure_optimizers(self):
        # Setup optimizer
        bias_group = []
        nonbias_group = []

        for param_name, param in self.salt.named_parameters():
            if 'bias' in param_name:
                bias_group.append(param)
            else:
                nonbias_group.append(param)

        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD([
                                            {'params': bias_group},
                                            {'params': nonbias_group, 'weight_decay': self.hparams.weight_decay,},
                                        ],
                                        lr=self.hparams.max_lr, 
                                        momentum=self.hparams.momentum,
                                        nesterov=True)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW([
                                            {'params': bias_group},
                                            {'params': nonbias_group, 'weight_decay': self.hparams.weight_decay,},
                                        ], 
                                        lr=self.hparams.max_lr, 
                                        # momentum=self.hparams.momentum,
                                        )
        else:
            raise ValueError('wrong optimizer option')
        
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                         T_max=20,
        #                                                         eta_min=self.hparams.min_lr,
        #                                                         verbose=True),
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.snapshot_size, T_mult=1, eta_min=self.hparams.min_lr),
            # 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'interval': 'epoch',
            'frequency': 1,
            # 'monitor': 'Loss/val',
        }
        # return optimizer
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False) 
        parser.add_argument('--model', default='res34v5', type=str, help='Model version')
        parser.add_argument('--loss_func', default='bce', type=str, help='Loss function')
        
        parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer')
        parser.add_argument('--snapshot_size', default=50, type=int, help='Number epochs per snapshot')
        parser.add_argument('--max_lr', default=0.01, type=float, help='max learning rate')
        parser.add_argument('--min_lr', default=0.001, type=float, help='min learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
        parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
        parser.add_argument('--save_pred', default='../predictions/', type=str, help='prediction save space')
        return parser

class ToLovaszHingeLossCB(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # print('Check', trainer.current_epoch)
        if trainer.current_epoch == 28:
            print('ToLovaszHingeLossCB')
            pl_module.criterion = LovaszHingeLoss()
            trainer.optimizers[0].param_groups[0]['lr'] *= 0.1
            print(trainer.optimizers[0].param_groups[0]['lr']) 

class ResetSnapshotCB(Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_train_epoch_start(self, trainer, pl_module):
        # print('Check', trainer.current_epoch)
        if trainer.current_epoch > 0 and (trainer.current_epoch + 1) % pl_module.hparams.snapshot_size == 0:
            trainer.optimizers[0].param_groups[0]['initial_lr'] /= 3
            trainer.optimizers[0].param_groups[0]['lr'] /= 3
            trainer.optimizers[0].param_groups[1]['initial_lr'] /= 3
            trainer.optimizers[0].param_groups[1]['lr'] /= 3
            init_lr = trainer.optimizers[0].param_groups[0]['initial_lr']
            trainer.lr_schedulers[0]['scheduler'].base_lrs = [init_lr, init_lr]

        if trainer.current_epoch > 0 and trainer.current_epoch % pl_module.hparams.snapshot_size == 0:
            print('Reset snapshot')
            trainer.checkpoint_callbacks[0].best_k_models = {}

def checkpointcb(args, checkpoint_dir):
    if args.save_checkpoint:
        return ModelCheckpoint(dirpath=checkpoint_dir,
                                save_top_k=1,
                                verbose=False,
                                monitor='precision',
                                mode='max',
                                filename='{epoch}-{precision:.4f}',
                                save_last=True)
    else: 
        return False

def get_logger(args):
    if args.logger_type == 'none':
        return False
    elif args.logger_type == 'wandb':
        logger = WandbLogger(project=f'TGS_Salt_Final',
                                entity='hiue', 
                                save_dir=args.log_dir,
                                id=f'{args.model}_{args.val_fold_idx}_{args.version}',
                                name=f'{args.model}_f{args.val_fold_idx}_v{args.version}')
    elif args.logger_type == 'tensorboard':
        logger = TensorBoardLogger(save_dir=args.log_dir,
                                version=f'version_{args.version}',
                                name='fold_{:02d}'.format(args.val_fold_idx))
    else:
        raise ValueError(f'Not support logger {args.logger} yet')
    return logger



def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--logger_type', type=str)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_checkpoint', type=str2bool)

    parser.add_argument('--swa_epoch_start', type=int, default=-1)


    parser = SaltDM.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitResUnet.add_model_specific_args(parser)
    return parser.parse_args(args)

def main(args, dm_setup='fit'):
    pl.seed_everything(args.seed)

    dm = SaltDM.from_argparse_args(args)
    dm.setup(dm_setup)
    model = LitResUnet(**vars(args))
    model.hparams.update(dm.kwargs)

    
    # print(wandb_logger.log_dir)
    
    # checkpoint_dir = os.path.join(tt_logger.log_dir, 'ckpt')

    checkpoint_dir = os.path.join('../params/{}/f{:02d}_{}'.format(args.model, args.val_fold_idx, args.version), 'ckpt')

    checkpoint_file_path = os.path.join(checkpoint_dir, 'last.ckpt')
    if os.path.isfile(checkpoint_file_path):
        args.resume_from_checkpoint = checkpoint_file_path
        print('Detect checkpoint:', args.resume_from_checkpoint)

    # Callbacks
    callbacks = []
    if args.loss_func == 'lovasz_hinge':
        callbacks.append(ToLovaszHingeLossCB())
    if args.logger_type != 'none':
        callbacks.append(LearningRateMonitor())
    if args.swa_epoch_start > 0:
        callbacks.append(StochasticWeightAveraging(swa_epoch_start=args.swa_epoch_start))
    if args.snapshot_size > 0:
        callbacks.append(ResetSnapshotCB(checkpoint_dir))

    trainer = pl.Trainer.from_argparse_args(args, 
                                            # logger=tt_logger, 
                                            logger=get_logger(args),
                                            callbacks=callbacks, 
#                                             callbacks=[ResetSnapshotCB(checkpoint_dir)], 
                                            checkpoint_callback=checkpointcb(args, checkpoint_dir))
    # trainer = pl.Trainer.from_argparse_args(args, logger=tt_logger, callbacks=[ResetSnapshotCB(), LearningRateMonitor()], checkpoint_callback=False)
    return dm, model, trainer

OPTIONS = {
    'v0': '''
    --augment_strategy 1
    --add_depth f
    --resize_pad f
    --num_workers 2
    --batch_size 16

    --optimizer adamw
    --snapshot_size 50
    --max_lr 3e-4
    --min_lr 1e-3
    --momentum 0.9
    --weight_decay 5e-4
    '''
}

if __name__ == '__main__':
    args = parse_args()
    dm, model, trainer = main(args)
    trainer.fit(model, dm)
