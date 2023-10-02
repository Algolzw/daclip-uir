import os
import cv2
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils


########### denoising ###############
def add_noise(tensor, sigma):
    sigma = sigma / 255 if sigma > 1 else sigma
    return tensor + torch.randn_like(tensor) * sigma


######## inpainting ###########
def mask_to(tensor, mask_root='data/datasets/gt_keep_masks/genhalf', mask_id=-1, n=100):
    batch = tensor.shape[0]
    if mask_id < 0:
        mask_id = np.random.randint(0, n, batch)
        masks = []
        for i in range(batch):
            masks.append(cv2.imread(os.path.join(mask_root, f'{mask_id[i]:06d}.png'))[None, ...] / 255.)
        mask = np.concatenate(masks, axis=0)
    else:
        mask = cv2.imread(os.path.join(mask_root, f'{mask_id:06d}.png'))[None, ...] / 255.

    mask = torch.tensor(mask).permute(0, 3, 1, 2).float()
    # for images are clipped or scaled
    mask = F.interpolate(mask, size=tensor.shape[2:], mode='nearest')
    masked_tensor = mask * tensor
    return masked_tensor + (1. - mask)

######## super-resolution ###########

def upscale(tensor, scale=4, mode='bicubic'):
    tensor = F.interpolate(tensor, scale_factor=scale, mode=mode)
    return tensor




