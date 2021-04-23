import os
from argparse import ArgumentParser
from types import SimpleNamespace
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
import pandas as pd

from .dataset import TGSSaltDataset
from .transforms import TGSTransform

from utils import str2bool

class SaltDM(pl.LightningDataModule):
    def __init__(self, data_root, num_workers, batch_size, is_pseudo, val_fold_idx, add_depth, resize_pad, augment_strategy):
        super().__init__()
        self.kwargs = {
            'data_root': data_root,
            'num_workers': num_workers,
            'batch_size': batch_size,
            'is_pseudo': is_pseudo,
            'val_fold_idx': val_fold_idx,
            'add_depth': add_depth,
            'resize_pad': resize_pad,
            'augment_strategy': augment_strategy,
        }
        self.hparams = SimpleNamespace(**self.kwargs)
        self.df = pd.read_csv(os.path.join(self.hparams.data_root, 'folds.csv'), index_col='id')
        depths = pd.read_csv(os.path.join(self.hparams.data_root, 'depths.csv'), index_col='id')
        self.test_df = depths.loc[~depths.index.isin(self.df.index)].reset_index()
 
    def setup(self, stage=None):
        # Assign train/val dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            df = self.df
            train_df = df.loc[df.fold != self.hparams.val_fold_idx].reset_index()
            val_df = df.loc[df.fold == self.hparams.val_fold_idx].reset_index()
            self.ds_train = TGSSaltDataset(self.hparams.data_root, train_df, 
                                            transforms=TGSTransform(augment_strategy=self.hparams.augment_strategy, 
                                                                    add_depth=self.hparams.add_depth, 
                                                                    resize_pad=self.hparams.resize_pad))
            self.ds_val = TGSSaltDataset(self.hparams.data_root, val_df, transforms=TGSTransform(augment_strategy=0, add_depth=self.hparams.add_depth, resize_pad=self.hparams.resize_pad)) # Không augmentation cho valid set
        elif stage == 'test':
            self.ds_test = TGSSaltDataset(self.hparams.data_root, self.test_df, image_set='test', transforms=TGSTransform(augment_strategy=0, add_depth=self.hparams.add_depth,resize_pad=self.hparams.resize_pad))

            
    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          shuffle=RandomSampler(self.ds_train),
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root', type=str, default='/ds/competition_data')
        
        parser.add_argument('--augment_strategy', type=int, help='Augmentation option')
        parser.add_argument('--val_fold_idx', type=int, help='Fold to train')
        parser.add_argument('--add_depth', type=str2bool, help='Add depth channels')
        parser.add_argument('--resize_pad', type=str2bool, help='Scale 2x and pad if need 256')
        parser.add_argument('--is_pseudo', default=False, type=bool, help='Use pseudolabels or not')

        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
        return parser