import torch
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image

def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    image = torch.zeros([3, h, w])
    image[0] = image_tensor
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, row, :] = const
    image[2] = image[0] * image[1]
    return image

def collate_mask_fn(batch):
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
    # disk = torch.FloatTensor(torch.ones(3,3).expand(1, 1 ,3, 3))
    return images, masks
    # loss_weights = torch.clamp(conv2d(masks, disk, padding=1), 0, 1) + torch.clamp(conv2d((1 - masks), disk, padding=1), 0, 1)
    # return images, masks, loss_weights

def collate_classify_fn(batch):
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
    masks = torch.cat(masks, dim=0)
    # disk = torch.FloatTensor(torch.ones(3,3).expand(1, 1 ,3, 3))
    return images, masks
    # loss_weights = torch.clamp(conv2d(masks, disk, padding=1), 0, 1) + torch.clamp(conv2d((1 - masks), disk, padding=1), 0, 1)
    # return images, masks, loss_weights

def generate_fold(root_ds, n_fold, outpath, img_size_ori=101):
    depths = pd.read_csv(os.path.join(root_ds, 'depths.csv'))
    depths.sort_values('z', inplace=True)
    # depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
    df_train = pd.read_csv(os.path.join(root_ds, 'train.csv'))

    df_train = df_train.merge(depths)

    df_train["images"] = [np.array(Image.open(os.path.join(root_ds, "train/images/{}.png".format(idx))), dtype=np.uint8) / 255 for idx in df_train.id]
    df_train["masks"] = [np.array(Image.open(os.path.join(root_ds, "train/masks/{}.png".format(idx))), dtype=np.uint8) / 255 for idx in df_train.id]
    df_train["coverage"] = df_train.masks.map(lambda x: np.sum(x) / pow(img_size_ori, 2))
    df_train.drop(['images', 'masks'], axis=1, inplace=True)

    dist = []
    for id in df_train.id.values:
        img = cv2.imread(os.path.join(root_ds, 'train', 'images', '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        dist.append(np.unique(img).shape[0])
    df_train['unique_pixels'] = dist

    df_train[['id', 'z', 'fold', 'unique_pixels', 'coverage']].sample(frac=1, random_state=123).to_csv(outpath, index=False)