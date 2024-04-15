import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
    import data.deg_util as deg_util
except ImportError:
    pass


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        GT_size = self.opt["patch_size"]
        LR_size = self.opt["patch_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # get LR image
        if self.LR_paths:  # LR exist
            LR_path = self.LR_paths[index]

            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            img_LR = util.read_img(self.LR_env, LR_path, resolution)
 
        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            # if img_GT.shape[0] != img_LR.shape[0]:
            #     img_GT = img_GT.transpose(1, 0, 2)
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate

            img_LR, img_GT = util.augment(
                [img_LR, img_GT],
                self.opt["use_flip"],
                self.opt["use_rot"],
                mode=self.opt["mode"],
            )

            # img_GT = deg_util.usm_sharp(img_GT)

            if random.random() < 0.2:
                img_GT = util.channel_convert(img_GT.shape[2], 'gray', [img_GT])[0]
                img_LR = util.channel_convert(img_LR.shape[2], 'gray', [img_LR])[0]

        # change color space if necessary
        if self.opt["color"]:
            img_LR = util.channel_convert(img_LR.shape[2], self.opt["color"], [img_LR])[0]
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]

        lq4clip = util.clip_transform(img_LR)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {"LQ": img_LR, "GT": img_GT, "LQ_clip": lq4clip, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)
