import os
import random
from datetime import datetime

import matplotlib
import numpy as np
from tokenizers.implementations import BaseTokenizer
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom


def colorize_text(tokens, scores):
    cmap = matplotlib.colormaps['coolwarm']
    token_colors = (matplotlib.colors.Normalize(vmin=-1, vmax=1)(scores))
    template = '<span class="attention-word"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''.join(
        [template.format(matplotlib.colors.rgb2hex(cmap(color)[:3]), word) for word, color in
         zip(tokens, token_colors)])
    return colored_string


def clean_tokens(words):
    replacements = {'▁': ' ', '&': '&amp;', '<': '&lt;', '>': '&gt;'}
    return [''.join(replacements.get(ch, ch) for ch in word) for word in words]


def get_patch_by_index(image: Image, index: int, processor, patch_dim=(24, 24), patch_size=(14, 14), img_side=336):
    image = processor.image_processor.resize(image, {'shortest_edge': img_side})
    image = processor.image_processor.center_crop(image, {'height': img_side, 'width': img_side})
    patch_dim_x, patch_dim_y = patch_dim
    patch_size_x, patch_size_y = patch_size
    row = index // patch_dim_y % patch_dim_x
    col = index % patch_dim_y
    patch = image[row * patch_size_x:(row + 1) * patch_size_x, col * patch_size_y:(col + 1) * patch_size_y]
    return patch


def plot_image_with_heatmap(image: Image, heatmap: np.array, ax: plt.Axes = None):
    image_np = np.array(image)
    zoom_y = image_np.shape[0] / heatmap.shape[0]
    zoom_x = image_np.shape[1] / heatmap.shape[1]
    heatmap_resized = zoom(heatmap, (zoom_y, zoom_x))
    if ax is None:
        plt.imshow(image_np, cmap='gray')
        plt.imshow(heatmap_resized, cmap='coolwarm', alpha=0.5)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(image_np, cmap='gray')
        ax.imshow(heatmap_resized, cmap='coolwarm', alpha=0.5)
        ax.axis('off')


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def timestamp():
    return datetime.now().strftime('%Y_%m_%d-%H_%M_%S')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


