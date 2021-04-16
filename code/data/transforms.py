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
import cv2

# main lib for augmentation
import albumentations as A

class HShear(A.DualTransform):
    """Transform for segmentation task."""
    def __init__(self, 
                limit_dx, 
                always_apply=False, 
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101):
        super(HShear, self).__init__(always_apply, p)
        self.limit_dx = limit_dx
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, image, dx=0, border_mode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR, **params):
        height, width = image.shape[:2]
        dx = int(dx*width)
        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ],np.float32)
        box1 = np.array([ [+dx,0], [width+dx,0],  [width-dx,height], [-dx,height], ],np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        image = cv2.warpPerspective(image, mat, (width,height),flags=interpolation,
                                    borderMode=border_mode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        return image

    def get_params(self):
        return {"dx" : np.random.uniform(*self.limit_dx)}

    def get_transform_init_args(self):
        return {
            "limit_dx": self.limit_dx,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
        }
# Các augmentation được sử dụng
# 1. 0.5 chance - hoz flip
# 2. 0.5 chance - choice one of:
#       2.1. random shift scale crop pad 0.2
#       2.2. do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.07, 0.07))
#       2.3. do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))
# 3. 0.5 chance - one of:
#       3.1. do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
#       3.2. do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))
class TGSTransform:
    """
        Augment and convert image and target to tensor. Add depth channel to image

        Params:
            augment - bool: perform augmentation if True
    """
    def __init__(self, augment, use_depth=False, image_orig_size=(101, 101)):
        self.augment = augment
        self.use_depth = use_depth
        # Define our augmentation pipeline.
        if augment:
            self.augtrans = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.RandomResizedCrop(width=image_orig_size[0], height=image_orig_size[1], p=1.0),
                    HShear((-0.07, 0.07), p=1.0),
                    A.Rotate(limit=(0, 15), p=1.0),
                ], p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, p=0.5)
            ])
        else:
            self.augtrans = A.NoOp()
        self.alb_trans = A.Compose([self.augtrans, A.Resize(128, 128)])
        
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
        # image = resize(image, (128, 128))
        # target = resize(target, (128, 128))\
        image = np.array(image)
        target = np.array(target)
        augmented = self.alb_trans(image=image, mask=target)
        image = to_tensor(augmented['image'])
        target = to_tensor(augmented['mask'])
        # Add depth channel
        # image = add_depth_channels(image)
        return image, target


class NetherlandsF3Transform:
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
                # iaa.Sometimes(
                #     0.5,
                #     iaa.size.CropToFixedSize(96, 96),
                #     iaa.Resize({"height": 96, "width": 96}),
                # ),
                iaa.Fliplr(0.5), # horizontal flips 50%
                # iaa.Flipud(0.5), # vertically flip 50%
            #     iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
            #     iaa.Sharpen((0.0, 1.0)),       # sharpen the image
                iaa.Sometimes(
                    0.5,
                    iaa.OneOf([
                        iaa.Crop(percent=(0, 0.2)),
                        iaa.Affine(
                            shear=(-0.07, 0.07)
                        ),
                        # iaa.Affine(
                        #     rotate=(-15, 15),
                        # ),

                    ])
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.OneOf([
                        iaa.arithmetic.Add((-26, 26)),
                        iaa.arithmetic.Multiply((1 - 0.08, 1 + 0.08))
                    ])
                ),
                # iaa.Resize({"height": 128, "width": 128}),
            ], random_order=False)
        else:
            self.seq = iaa.Sequential([
                # iaa.size.CropToFixedSize(64, 64, position="center"),
                # iaa.Resize({"height": 128, "width": 128})
            ], random_order=False)
        
    def __call__(self, image, target):
        """
        Params:
            image - PIL Image
            target - PIL Image: segmentation map
            depth - float
    
        Returns:
            image - [2, W, H]
            target - [1, W, H]
        """
        # image = resize(image, (128, 128))
        # target = resize(target, (128, 128))\
        image = np.array(image)
        image_aug = self.seq(image=image)
        image = to_tensor(image_aug.copy())
        target = torch.LongTensor([target])
        # image = add_depth_channels(image)
        return image, target


#         self.seq = iaa.Sequential([
#     # iaa.Sometimes(
#     #     0.5,
#     #     iaa.size.CropToFixedSize(96, 96),
#     #     iaa.Resize({"height": 96, "width": 96}),
#     # ),
#     iaa.Fliplr(0.5), # horizontal flips 50%
#     # iaa.Flipud(0.5), # vertically flip 50%
# #     iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
# #     iaa.Sharpen((0.0, 1.0)),       # sharpen the image
#     iaa.Sometimes(
#         0.5,
#         iaa.OneOf([
#             iaa.Crop(percent=(0, 0.2)),
#             iaa.Affine(
#                 shear=(-0.07, 0.07)
#             ),
#             iaa.Affine(
#                 rotate=(-15, 15),
#             ),

#         ])
#     ),
#     iaa.Sometimes(
#         0.5,
#         iaa.OneOf([
#             iaa.arithmetic.Add((-26, 26)),
#             iaa.arithmetic.Multiply((1 - 0.08, 1 + 0.08))
#         ])
#     ),
#     iaa.Resize({"height": 128, "width": 128}),
# ], random_order=False)