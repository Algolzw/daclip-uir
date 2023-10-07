import os
import random
import sys

import numpy as np
import pandas as pd

import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

DEGRADATION_TYPES = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def get_paired_paths(dataroot):
    """
    Read LQ (Low Quality) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """
    GT_paths, LQ_paths, dagradations = [], [], []
    for deg_type in DEGRADATION_TYPES:
        paths1 = _get_paths_from_images(os.path.join(dataroot, deg_type, 'GT'))
        paths2 = _get_paths_from_images(os.path.join(dataroot, deg_type, 'LQ'))

        GT_paths.extend(paths1)  # GT list
        LQ_paths.extend(paths2)  # LR list

        dagradations.extend([deg_type]*len(paths2))
    print(f'GT length: {len(GT_paths)}, LQ length: {len(LQ_paths)}')
    return GT_paths, LQ_paths, dagradations


def generate_captions(dataroot, ci, mode='train'):
    GT_paths, LQ_paths, dagradations = get_paired_paths(os.path.join(dataroot, mode))

    future_df = {"filepath":[], "title":[]}
    for gt_image_path, lq_image_path, dagradation in tqdm(zip(GT_paths, LQ_paths, dagradations)):
        image = Image.open(gt_image_path).convert('RGB')
        caption = ci.generate_caption(image)
        title = f'{caption}: {dagradation}'

        future_df["filepath"].append(lq_image_path)
        future_df["title"].append(title)

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(dataroot, f"daclip_{mode}.csv"), index=False, sep="\t"
    )


if __name__ == "__main__":
    dataroot = 'datasets/universal'

    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    generate_captions(dataroot, ci, 'val')
    generate_captions(dataroot, ci, 'train')
    

