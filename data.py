from typing import Any, Callable, Optional
from torchvision.datasets import VisionDataset, VOCDetection
# import tifffile as tiff
import os
from torchvision.transforms.functional import to_tensor, hflip, vflip, resize
import pandas as pd
from PIL import Image
import glob
import torch
from torch.nn.functional import conv2d
import random
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

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
        if image_set == 'train':
            self.df = df
            self.df = pd.merge( self.df, pd.read_csv(os.path.join(self.root, 'depths.csv')) , left_on='id', right_on='id', how='inner')
            depths = self.df['z']
            self.df['z'] = (depths - depths.min()) / (depths.max() - depths.min())
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
        return img, target
    
    def __len__(self) -> int:
        return len(self.image_ids)

class TGSTransform:
    """
        Augment and convert image and target to tensor. Add depth channel to image

        Params:
            augment - bool: perform augmentation if True
    """
    def __init__(self, augment, use_depth=False):
        self.augment = augment
        self.use_depth = use_depth
        # Define our augmentation pipeline.
        if augment:
            self.seq = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips 50%
                iaa.Flipud(0.5), # vertically flip 50%
            #     iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
            #     iaa.Sharpen((0.0, 1.0)),       # sharpen the image
                # iaa.Sometimes(
                #     0.5,
                #     iaa.Affine(
                #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #         rotate=(-45, 45),
                #         shear=(-8, 8)
                #     )
                # ),
                # iaa.Sometimes(
                #     0.5,
                #     iaa.ElasticTransformation(alpha=(0, 5.0), sigma=(0.5, 1.0))  # apply water effect (affects segmaps)
                # ),
            ], random_order=False)
        
    def __call__(self, image, target, depth):
        """
        Params:
            image - PIL Image
            target - PIL Image: segmentation map
            depth - float
    
        Returns:
            image - [2, W, H]
            target - [1, W, H]
        """
        image = resize(image, (128, 128))
        target = resize(target, (128, 128))
        if self.augment:
            # Augment
            image = np.array(image)
            target = np.array(target)
            target[target > 1] = 1
            segmap = SegmentationMapsOnImage(target, shape=image.shape)
            image_aug, segmap_aug = self.seq(image=image, segmentation_maps=segmap)
            image = to_tensor(image_aug.copy())
            target = torch.FloatTensor(segmap_aug.get_arr()).unsqueeze(0)
        else:
            image = to_tensor(image)
            target = to_tensor(target)

        # Add depth channel
        if self.use_depth:
            image = torch.cat((image, torch.ones_like(image)*depth), dim=0)
        return image, target


    
def collate_fn(batch):
    """
    Params: each sample in batch includes
        0 - image: [2, W, H]
        1 - target:[1, W, H]
    Returns:
        images: [bs, 2, W, H]
        target: [bs, 1, W, H]
        loss_weights: [bs, 1, W, H]
    """
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    disk = torch.FloatTensor(torch.ones(3,3).expand(1, 1 ,3, 3))
    return images, masks
    # loss_weights = torch.clamp(conv2d(masks, disk, padding=1), 0, 1) + torch.clamp(conv2d((1 - masks), disk, padding=1), 0, 1)
    # return images, masks, loss_weights