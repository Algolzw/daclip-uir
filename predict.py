# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import cv2
import argparse
import sys, os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
import torchvision.utils as tvutils

sys.path.insert(0, "universal-image-restoration")
sys.path.insert(0, "universal-image-restoration/config/daclip-sde")
import options as option
from models import create_model


import open_clip
import utils as util

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        config_file = "universal-image-restoration/config/daclip-sde/options/test.yml"
        opt = option.parse(config_file, is_train=False)
        opt = option.dict_to_nonedict(opt)

        # download daclip_ViT-B-32.pt to ./pretrained first
        self.model = create_model(opt)
        self.device = self.model.device

        self.clip_model, preprocess = open_clip.create_model_from_pretrained(
            "daclip_ViT-B-32", pretrained=opt["path"]["daclip"]
        )
        self.clip_model = self.clip_model.to(self.device)

        self.sde = util.IRSDE(
            max_sigma=opt["sde"]["max_sigma"],
            T=opt["sde"]["T"],
            schedule=opt["sde"]["schedule"],
            eps=opt["sde"]["eps"],
            device=self.device,
        )
        self.sde.set_model(self.model.model)

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        image = cv2.imread(str(image))
        image = image[:, :, [2, 1, 0]] / 255.0
        img4clip = clip_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_context, degra_context = self.clip_model.encode_image(
                img4clip, control=True
            )
            image_context = image_context.float()
            degra_context = degra_context.float()

        LQ_tensor = (
            torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        )
        noisy_tensor = self.sde.noise_state(LQ_tensor)
        self.model.feed_data(
            noisy_tensor,
            LQ_tensor,
            text_context=degra_context,
            image_context=image_context,
        )
        self.model.test(self.sde)
        visuals = self.model.get_current_visuals(need_GT=False)
        output = util.tensor2img(visuals["Output"].squeeze())

        out_path = "/tmp/out.png"

        cv2.imwrite(out_path, output)

        return Path(out_path)


def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose(
        [
            Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(resolution),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )(pil_image)
