import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

sys.path.append("..")
import data.util as util
import data.deg_util as deg_util

class MDGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.GT_paths = None
        self.GT_size = opt["patch_size"]

        self.GT_paths = util.get_image_paths(
            opt["data_type"], opt["dataroot_GT"]
        )  # GT list
        assert self.GT_paths, "Error: GT paths are empty."

        self.random_scale_list = [1]

    def __getitem__(self, index):
        GT_size = self.GT_size

        # get GT image
        GT_path = self.GT_paths[index]
        img_GT = util.read_img( None, GT_path, None)  # return: Numpy float32, HWC, BGR, [0,1]

        # change color space if necessary
        if self.opt["color"]:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[0]

        if self.opt["phase"] == "train":
            # if GT_size < 512:
            #     img_GT = cv2.resize(img_GT, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA) # INTER_CUBIC, INTER_AREA
            #     if img_GT.ndim == 2: img_GT = np.expand_dims(img_GT, axis=2)

            H, W, C = img_GT.shape
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_GT = util.augment(
                img_GT,
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )
            if random.random() < 0.1:
                img_GT = util.channel_convert(img_GT.shape[2], 'gray', [img_GT])[0]
                img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[0]

        img_LQ = deg_util.random_degrade(img_GT)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        lq4clip = util.clip_transform(img_LQ)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        return {"GT": img_GT, "LQ": img_LQ, "LQ_clip": lq4clip, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)
