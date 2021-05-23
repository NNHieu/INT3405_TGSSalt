import torch
from model import get_model
import time
from torchvision.transforms.functional import resize
from utils import rle_encode
import pandas as pd
import argparse
import os
from data.dataset import TGSSaltDataset
from data.transforms import TGSTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

def pred_batch(model, images, tta_flip):
    logit_null = model(images)
    preds_null = torch.sigmoid(logit_null).squeeze(1).detach()
    if tta_flip:
        logit_flip = model(images.flip(-1))
        preds_flip = torch.sigmoid(logit_flip).squeeze(1).flip(-1).detach()
        return (preds_flip + preds_null) / 2
    return preds_null

def pred_all(model, test_dl, tta_flip, is_valid=False):
    preds_test_upsampled = []
    model.eval()
    model.cuda()
    with torch.no_grad():
        for batch in tqdm(test_dl):
            if is_valid:
                images = batch[0]
            else:
                images = batch
            pred_masks = pred_batch(model, images.cuda(), tta_flip)
            preds_test_upsampled.append(pred_masks)
    return preds_test_upsampled

def resize_to_orig(preds_test):
    preds_101 = torch.zeros(preds_test.shape[0], 101, 101)
    for idx in range(preds_test.shape[0]):
        preds_101[idx] = resize(preds_test[idx], (101, 101))
    return preds_101

def build_summission(preds_101, image_ids):
    t1 = time.time()
    pred_dict = {idx: rle_encode(preds_101[i]) for i, idx in enumerate(image_ids)}
    t2 = time.time()
    print(f"Encode time = {t2-t1} s")
    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns =['rle_mask']
    return sub



def submit_model(model, data_root, add_depth, resize_pad, tta_flip, batch_size=16, num_workers=2, pred_threshold=0.5):
    df = pd.read_csv(os.path.join(data_root, 'folds.csv'), index_col='id')
    depths = pd.read_csv(os.path.join(data_root, 'depths.csv'), index_col='id')
    test_df = depths.loc[~depths.index.isin(df.index)].reset_index()
    ds_test = TGSSaltDataset(data_root, test_df, image_set='test', transforms=TGSTransform(augment_strategy=0, add_depth=add_depth,resize_pad=resize_pad))
    test_dl = DataLoader(ds_test,
                          shuffle=False,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=True)
    preds_test = pred_all(model, test_dl, tta_flip)
    preds_test = torch.cat(preds_test, dim=0).unsqueeze(1)
    preds_101 = resize_to_orig(preds_test)
    preds_101 = (preds_101.cpu().numpy() > pred_threshold).astype(int)
    sub = build_summission(preds_101, ds_test.image_ids)
    return sub
