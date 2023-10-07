import os
import torch
from PIL import Image
import open_clip
from tqdm import tqdm


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path):
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

checkpoint = 'logs/daclip_ViT-B-32_b768x4_lr3e-5_e50/checkpoints/epoch_50.pt'

model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

val_root = 'datasets/universal/val'
degradations = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']

text = tokenizer(degradations)
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)


for i, degra in enumerate(degradations):
    root_path = os.path.join(val_root, degra, 'LQ')
    image_paths = get_paths_from_images(root_path)
    acc = 0.0
    for im_path in tqdm(image_paths):
        image = preprocess(Image.open(im_path)).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, image_features = model.encode_image(image, control=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            index = torch.argmax((image_features @ text_features.T)[0])
            acc += float(index == i)
    acc /= len(image_paths)
    print(f'degradation: {degra},\t accuracy: {acc:.6f}')


