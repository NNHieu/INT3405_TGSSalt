import matplotlib.pyplot as plt
import numpy as np

def plot_mask_on_img(ax, img, mask, meta=None, y_max=101, x_max=101):
    mask = np.ma.masked_where( mask == 0, mask)
    ax.imshow(img, cmap="gray")
    ax.imshow(mask, alpha=0.5, cmap="jet")
    if meta is not None:
        ax.text(1, y_max-1, meta['z'], color="white")
        ax.text(x_max - 1, 1, round(meta['coverage'], 2), color="white", ha="right", va="top")
        ax.text(1, 1, meta['coverage_class'], color="white", ha="left", va="top")
    ax.set_yticklabels([])
    ax.set_xticklabels([])