import torch

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