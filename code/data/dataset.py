from typing import Any, Callable, Optional
from torchvision.datasets import VisionDataset, VOCDetection
from torch.utils.data import Dataset
import os
from torchvision.transforms.functional import to_tensor, hflip, vflip, resize
import pandas as pd
from PIL import Image
import glob
import torch
from torch.nn.functional import conv2d
import random
import numpy as np

class TGSSaltDataset(VisionDataset):
    def __init__(self, 
            root: str,
            df,
            image_set: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,) -> None:
        super(TGSSaltDataset, self).__init__(root, transform=transform, target_transform=target_transform, transforms=transforms)
        self.image_set = image_set
        self.df = df
        # self.df = pd.merge( self.df, pd.read_csv(os.path.join(self.root, 'depths.csv')) , left_on='id', right_on='id', how='inner')
        # depths = self.df['z']
        # self.df['z'] = (depths - depths.min()) / (depths.max() - depths.min())
        if image_set == 'train':
            self.image_ids = self.df['id']
            self.image_path = os.path.join(self.root, 'train')
        elif image_set == 'test':
            self.image_ids = [x[:-4] for x in os.listdir(os.path.join(self.root, 'test', 'images'))]
            self.image_path = os.path.join(self.root, 'test')

    def __getitem__(self, index) -> Any:
        img = Image.open(os.path.join(self.image_path, 'images', self.image_ids[index] + '.png')).convert('L')
        target = None
        if self.image_set == 'train':
            target = Image.open(os.path.join(self.image_path, 'masks', self.image_ids[index] + '.png')).convert('L')
        if self.transforms is not None:
            img, target = self.transforms(img, target, self.df.loc[index]['z'])
        if target is None:
            return img
        return img, target
    
    def __len__(self) -> int:
        return len(self.image_ids)


class NetherlandsF3DS(Dataset):
    def __init__(self, root, df, transforms=None):
        super().__init__()
        self.root = root
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.df.name[index] + '.png'))
        label = self.df['label'][index]
        if self.transforms is not None:
            img, label = self.transforms(img, label)
        return img, label
    
    def __len__(self) -> int:
        return len(self.df.index)