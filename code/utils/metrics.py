import numpy as np
import torch

def cal_mAP(predict,truth, pred_threshold=0.5):

    # N = len(predict)
    # predict = predict.view(N,-1)
    # truth   = truth.view(N,-1)

    predict = predict>pred_threshold
    truth   = truth>0.5
    intersection = truth & predict # Intersection
    union        = truth | predict # Union
    iou = torch.sum(intersection, dim=[-1, -2])/(torch.sum(union, dim=[-1, -2]) +1e-8)
    #-------------------------------------------
    result = []
    precision = []
    is_empty_truth   = (torch.sum(truth, dim=[-1, -2])==0)
    is_empty_predict = (torch.sum(predict, dim=[-1, -2])==0)

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou>=t

        tp  = torch.sum((~is_empty_truth)  & (~is_empty_predict) & (iou> t))
        fp  = torch.sum((~is_empty_truth)  & (~is_empty_predict) & (iou<=t))
        fn  = torch.sum((~is_empty_truth)  & ( is_empty_predict))
        fp_empty = torch.sum(( is_empty_truth)  & (~is_empty_predict))
        tn_empty = torch.sum(( is_empty_truth)  & ( is_empty_predict))

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)
        # print(p)
        # result.append( torch.cat((tp,fp,fn,tn_empty,fp_empty), dim=-1))
        precision.append(p)

    # result = torch.cat(result).permute(1,2,0)
    precision = torch.tensor(precision)
    precision = torch.mean(precision)

    # return precision, result, threshold
    return precision

def cal_mIoU(masks, preds, threshold_range=(0.5, 1, 0.05), reduce_batch=True):
    """
    Params:
        masks [bs, 1, W, H]
        pred [bs, 1, W, H]
    
    Return:
        mIoU: float
    """
    batch_size = masks.shape[0]
    metric = []
    thresholds = torch.arange(*threshold_range)
    for thresh in thresholds:
        t, p = masks > 0, preds > thresh
        I = torch.logical_and(t, p)
        U = torch.logical_or(t, p)
        iou = (torch.sum(I, dim=[-1, -2]) + 1e-10) / (torch.sum(U, dim=[-1, -2]) + 1e-10)
        iou = iou.squeeze()
        if reduce_batch:
            metric.append(torch.mean(iou))
        else:
            metric.append(iou)
    return torch.mean(torch.stack(metric, dim=0), dim=0)