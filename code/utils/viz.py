import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

def plot_training_samples(imgs, masks, meta=None, y_max=101, x_max=101, figsize=(16, 5), mask_over_image=False, bl_texts=None, subtitle=None):
    assert len(imgs) == len(masks)
    n = len(imgs)
    r = 1 if mask_over_image else 2
    fig, axs = plt.subplots(r, n, figsize=figsize)
    w, h = imgs[0].shape[-2], imgs[0].shape[-1]
    if subtitle is not None: fig.suptitle(subtitle, y=0.75)
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        if len(img.shape) == 3 and isinstance(img, Tensor):
            img = img.squeeze()
        if len(mask.shape) == 3 and isinstance(mask, Tensor):
            mask = mask.squeeze()

        if mask_over_image:
            plot_mask_on_img(axs[i], img, np.array(mask))
            if bl_texts is not None:
                axs[i].text(1, h - 1, bl_texts[i], color="white")

        else:
            axs[0, i].imshow(img, cmap='gray')
            axs[1, i].imshow(mask, cmap='gray')
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
        

def plot_mask_on_img(ax, img, mask, meta=None, y_max=101, x_max=101):
    mask = mask + 1
    mask = np.ma.masked_where( mask == 1, mask)
    ax.imshow(img, cmap="gray")
    ax.imshow(mask == 0, alpha=0.5, cmap="jet")
    if meta is not None:
        ax.text(1, y_max-1, meta['z'], color="white")
        ax.text(x_max - 1, 1, round(meta['coverage'], 2), color="white", ha="right", va="top")
        ax.text(1, 1, meta['coverage_class'], color="white", ha="left", va="top")
    ax.set_yticklabels([])
    ax.set_xticklabels([])