import numpy as np
import torch

def mAP(predict,truth, threshold=0.5):

    N = len(predict)
    predict = predict.reshape(N,-1)
    truth   = truth.reshape(N,-1)

    predict = predict>threshold
    truth   = truth>0.5
    intersection = truth & predict # Intersection
    union        = truth | predict # Union
    iou = intersection.sum(1)/(union.sum(1)+1e-8)
    print(iou)
    #-------------------------------------------
    result = []
    precision = []
    is_empty_truth   = (truth.sum(1)==0)
    is_empty_predict = (predict.sum(1)==0)

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou>=t

        tp  = (~is_empty_truth)  & (~is_empty_predict) & (iou> t)
        fp  = (~is_empty_truth)  & (~is_empty_predict) & (iou<=t)
        fn  = (~is_empty_truth)  & ( is_empty_predict)
        fp_empty = ( is_empty_truth)  & (~is_empty_predict)
        tn_empty = ( is_empty_truth)  & ( is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append( np.column_stack((tp,fp,fn,tn_empty,fp_empty)) )
        precision.append(p)

    result = np.array(result).transpose(1,2,0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold

def mIoU(masks, preds, threshold_range=(0.5, 1, 0.05)):
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
        metric.append(torch.mean(iou))
    return torch.mean(torch.FloatTensor(metric))