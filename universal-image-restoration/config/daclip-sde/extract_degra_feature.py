import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

# import open_clip

import options as option
from models import create_model
from tqdm import tqdm

sys.path.insert(0, "../../")
import open_clip
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = '/root/workplace/daclip-uir/dataset/degra_feature/'
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")

    torch.backends.cudnn.benchmark = True
    resume_state = None


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)

    assert train_loader is not None
    assert val_loader is not None


    # clip_model, _preprocess = clip.load("ViT-B/32", device=device)
    if opt['path']['daclip'] is not None:
        clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
    else:
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model = clip_model.to(device)

    j=0
    for _, train_data in tqdm(enumerate(train_loader)):
        img4clip, degra_type = train_data["LQ_clip"].to(device), train_data["type"]
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, degra_context = clip_model.encode_image(img4clip, control=True)
            degra_context = degra_context.float()
            #degra_contex [B, 512]
            feature = degra_context.cpu().detach().numpy()
            #iterative over first axis of numpy array(feature)
            for i in range(feature.shape[0]):
                degra_type_i = degra_type[i]
                feature_i = feature[i]
                feature_i_unsqueezed = np.expand_dims(feature_i, axis=0)
                np.save(f'{base_name}{degra_type_i}_{j}'+".npy", feature_i_unsqueezed)
                j += 1

if __name__ == "__main__":
    main()
