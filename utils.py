import torch

def calculate_mAP(masks, preds, threshold_range=(0.5, 1, 0.05)):
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

    # for batch_idx in range(batch_size):
    #     t, p = masks[batch_idx]>0, preds[batch_idx]>0
    #     intersection = torch.logical_and(t, p)
    #     union = torch.logical_or(t, p)
    #     iou = (torch.sum(intersection > 0) + 1e-10 )/ (torch.sum(union > 0) + 1e-10)
    #     thresholds = torch.arange(0.5, 1, 0.05)
    #     s = []
    #     for thresh in thresholds:
    #         s.append(iou > thresh)
    #     metric.append(torch.mean(torch.FloatTensor(s)))

    # return torch.mean(torch.FloatTensor(metric))
