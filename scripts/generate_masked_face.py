import cv2
import numpy as np
import os

def add_random_mask(img, size=None, mask_root='inpainting_masks', mask_id=-1, n=100):
    if mask_id < 0:
        mask_id = np.random.randint(n)

    mask = cv2.imread(os.path.join(mask_root, f'{mask_id:06d}.png')) / 255.
    if size is None:
        mask = cv2.resize(mask, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)
    else:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        rnd_h = np.random.randint(0, max(0, size[0] - img.shape[0]))
        rnd_w = np.random.randint(0, max(0, size[1] - img.shape[1]))
        mask = mask[rnd_h : rnd_h + img.shape[0], rnd_w : rnd_w + img.shape[1]]

    return mask * img + (1. - mask)

target_dir = '.'
source_dir = '../images/'
im_name = 'Elon-Musk-256x256.jpg'
im = cv2.imread(os.path.join(source_dir, im_name)) / 255.
masked_im = add_random_mask(im) * 255
cv2.imwrite(os.path.join(target_dir, im_name), masked_im)