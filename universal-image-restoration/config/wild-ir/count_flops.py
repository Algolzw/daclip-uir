import argparse
import sys 
import torch
from torchsummaryX import summary

import options as option
from models import create_model

sys.path.insert(0, "../../")
import open_clip

parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/inference.yml",
    help="Path to option YMAL file of Predictor.",
)
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)
model = create_model(opt)
device = model.device

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-L-14', pretrained=opt['path']['daclip'])
clip_model = clip_model.cuda()

test_tensor = torch.randn(1, 3, 256, 256).cuda()
clip_tensor = torch.randn(1, 3, 224, 224).cuda()

summary(clip_model, clip_tensor)
with torch.no_grad(), torch.cuda.amp.autocast():
    image_context, degra_context = clip_model.encode_image(clip_tensor, control=True)
    image_context = image_context.float()
    degra_context = degra_context.float()

summary(model.model, test_tensor, test_tensor, 1, image_context, image_context)