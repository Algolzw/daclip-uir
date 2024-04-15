import argparse
import logging
import os.path
import sys
import time

import numpy as np
import torch
from IPython import embed
from tqdm import tqdm

import options as option
from models import create_model

sys.path.insert(0, "../../")
import open_clip
import utils as util
from data import create_dataloader, create_dataset

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-L-14', pretrained=opt['path']['daclip'])
clip_model = clip_model.to(device)

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
sampling_mode = opt["sde"]["sampling_mode"]

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_times = []

    for i, test_data in enumerate(test_loader):
        # if i > 1200: break
        # if i <= 1200: continue
        print(i)
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ = test_data["LQ"]
        img4clip = test_data["LQ_clip"].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_context, degra_context = clip_model.encode_image(img4clip, control=True)
            image_context = image_context.float()
            degra_context = degra_context.float()

        noisy_state = sde.noise_state(LQ)
        model.feed_data(noisy_state, LQ, GT=None, text_context=degra_context, image_context=image_context)
        tic = time.time()
        model.test(sde, mode=sampling_mode, save_states=False)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals(need_GT=False)
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        mode = "gray" if test_data["gray"] == True else "RGB"

        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(output, save_img_path, mode=mode)

        # LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        # LQ_img_path = os.path.join(dataset_dir, img_name + "_LQ.png")
        # util.save_img(LQ_, LQ_img_path)

    print(f"average test time: {np.mean(test_times):.4f}")


