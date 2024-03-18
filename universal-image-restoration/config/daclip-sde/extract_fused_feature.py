import argparse
import logging
import math
import os
import random
import sys
import copy
import GPUtil
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

#### options
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

#### Create test dataset and dataloader
for phase, dataset_opt in opt["datasets"].items():
    if phase == "train":
        train_set = create_dataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
        total_iters = int(opt["train"]["niter"])
        total_epochs = int(math.ceil(total_iters / train_size))
        train_sampler = None
        train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)

# load pretrained model by default
model = create_model(opt)
device = model.device

# clip_model, _preprocess = clip.load("ViT-B/32", device=device)
if opt['path']['daclip'] is not None:
    clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
else:
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model = clip_model.to(device)

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

scale = opt['degradation']['scale']
base_path = '/home/viplab/dataset/DA-CLIP/degra_feature/'
j = 0
for _, train_data in tqdm(enumerate(train_loader)):
    img_path = train_data["LQ_path"]
    #### input dataset_LQ
    LQ, GT, degra_type = train_data["LQ"], train_data["GT"], train_data["type"]
    visual_feature = train_data["visual_feature"]
    img4clip = train_data["LQ_clip"].to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        image_context = image_context.float()
        degra_context = degra_context.float()

    noisy_state = sde.noise_state(LQ)
    model.feed_data(noisy_state, LQ, visual_feature, GT, text_context=degra_context, image_context=image_context)
    with torch.no_grad():
        degra_feature = model.get_feature(sde, save_states=False)
    for i in range(degra_feature.shape[0]):
        degra_type_i = degra_type[i]
        degra_feature_i = degra_feature[i]
        degra_feature_i = degra_feature_i.cpu().numpy()
        np.save(f'{base_path}{degra_type_i}_{j}'+"_feature_fusion.npy", degra_feature_i)
        j += 1