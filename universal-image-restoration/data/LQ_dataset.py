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
except ImportError:
    pass


class LQDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and LR image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LQ_paths = None
        self.LR_env = None  # environment for lmdb
        self.LR_size = opt["LR_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LQ_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
        elif opt["data_type"] == "img":
            self.LQ_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.LQ_paths, "Error: LQ paths are empty."

        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.LR_env = lmdb.open(
            self.opt["dataroot_LR"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.LR_env is None:
                self._init_lmdb()

        LR_path = None
        scale = self.opt["scale"]
        LR_size = self.opt["LR_size"]

        # get LR image
        LR_path = self.LQ_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.LR_sizes[index].split("_")]
        else:
            resolution = None
        img_LR = util.read_img(
            self.LR_env, LR_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_LR = util.modcrop(img_LR, scale)

        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape

            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]

            # augmentation - flip, rotate
            img_LR = util.augment(
                img_LR,
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

        # change color space if necessary
        if self.opt["color"]:
            img_LR = util.channel_convert(img_LR.shape[2], self.opt["color"], [img_LR])[
                0
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        return {"LQ": img_LR, "LQ_path": LR_path}

    def __len__(self):
        return len(self.LQ_paths)
