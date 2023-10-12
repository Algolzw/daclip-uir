
## Controlling Vision-Language Models for Universal Image Restoration <br><sub>Official PyTorch Implementation of DA-CLIP. </sub>

[Project Page](https://algolzw.github.io/daclip-uir) | [Paper](https://arxiv.org/abs/2310.01018)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/daclip-uir-colab/blob/main/daclip_uir_gradio_colab.ipynb) [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/fffiloni/DA-CLIP)

![daclip](figs/teaser.jpg)

### Overview framework:

![daclip](figs/overview.jpg)

### Updates

[**2023.10.11**] We updated an online demo in [Colab](https://colab.research.google.com/github/camenduru/daclip-uir-colab/blob/main/daclip_uir_gradio_colab.ipynb). Thanks to [@camenduru](https://github.com/camenduru)!<br>
[**2023.10.11**] We updated the [Gradio file](https://github.com/Algolzw/daclip-uir/tree/main/universal-image-restoration/config/daclip-sde/app.py) with example LQ images and provide more examples from our test dataset [here](https://drive.google.com/file/d/1C1nmP5kJXzxrULxTMVWF5P30qezqP6kn/view?usp=sharing). We also thank [@fffiloni](https://github.com/fffiloni) for sharing the Hugging Face [demo](https://huggingface.co/spaces/fffiloni/DA-CLIP) 🔥!!!<br>
[**2023.10.09**] The **pretrained weights** of DA-CLIP and the Universal IR model are released in [link1](https://drive.google.com/file/d/1A6u4CaVrcpcZckGUNzEXqMF8x_JXsZdX/view?usp=sharing) and [link2](https://drive.google.com/file/d/1eXsyrmAbWOvhIY4Wbt5v4IxaggA5aZMG/view?usp=sharing), respectively. In addition, we also provide a [Gradio](https://gradio.app/) app file for the case that you want to [test your own images](https://github.com/Algolzw/daclip-uir#Gradio).<br>

#### Notice!!
In testing we found that the current pretrained model is still difficult to process some real-world images 🙁 which might have distribution shifts with our training dataset (captured from different devices or with different resolutions or degradations). We regard it as a future work and will try to make our model more practical! We also encourage users who are interested in our work to train their own models with larger dataset and more degradation types.

BTW, **we also found that directly resizing input images will lead a poor performance for most tasks**. We could try to add the resize step into the training but it always destroys the image quality due to interpolation.

For the inpainting task our current only support face inpainting due to the [dataset limitation](https://github.com/Algolzw/daclip-uir/issues/8#issuecomment-1759528246). We provide our mask examples [here](https://drive.google.com/file/d/1U91P0w993k4emczMSImakzY1RTooyZez/view?usp=sharing) and you can use the [add\_random\_mask](https://github.com/Algolzw/daclip-uir/blob/00b4b6e9960f6ca712d7ef7362ba0348b5613eb4/universal-image-restoration/data/deg_util.py#L783) function to create uncompleted faces.


## How to Run the Code?


### Dependenices

* OS: Ubuntu 20.04
* nvidia:
	- cuda: 11.4
* python 3.8

### Install

We advise you first create a virtual environment with:

```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt

```

### DA-CLIP Usage

Get into the `universal-image-restoration` directory and run:

```python
import torch
from PIL import Image
import open_clip

checkpoint = 'pretrained/daclip_ViT-B-32.pt'
model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("haze_01.png")).unsqueeze(0)
degradations = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']
text = tokenizer(degradations)

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    image_features, degra_features = model.encode_image(image, control=True)
    degra_features /= degra_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * degra_features @ text_features.T).softmax(dim=-1)
    index = torch.argmax(text_probs[0])

print(f"Task: {task_name}: {degradations[index]} - {text_probs[0][index]}")
```

### Dataset Preparation

Preparing the train and test datasets following our paper Dataset Construction section as:

```bash
#### for training dataset ####
#### (uncompleted means inpainting) ####
datasets/universal/train
|--motion-blurry
|  |--LQ/*.png
|  |--GT/*.png
|--hazy
|--jpeg-compressed
|--low-light
|--noisy
|--raindrop
|--rainy
|--shadowed
|--snowy
|--uncompleted

#### for testing dataset ####
#### (the same structure as train) ####
datasets/universal/val
...

#### for clean captions ####
datasets/universal/daclip_train.csv
datasets/universal/daclip_val.csv
```

Then get into the `universal-image-restoration/config/daclip-sde` directory and modify the dataset paths in option files in `options/train.yml` and `options/tes.yml`. 

You can add more tasks or datasets to both `train` and `val` directories and add the degradation word to `distortion`.


### Training

#### DA-CLIP 
See [DA-CLIP.md](da-clip/README.md) for details.

#### Universal Image Restoration
The main code for training is in `universal-image-restoration/config/daclip-sde` and the core network for DA-CLIP is in `universal-image-restoration/open_clip/daclip_model.py`.

* Put the pretrained [**DA-CLIP weights**](https://drive.google.com/file/d/1A6u4CaVrcpcZckGUNzEXqMF8x_JXsZdX/view?usp=sharing) to `pretrained` directory and check the `daclip` path.

* You can then train the model following below bash scripts:

```bash
cd universal-image-restoration/config/daclip-sde

# For single GPU:
python3 train.py -opt=options/train.yml

# For distributed training, need to change the gpu_ids in option file
python3 -m torch.distributed.launch --nproc_per_node=2 --master_poer=4321 train.py -opt=options/train.yml --launcher pytorch
```

The models and training logs will save in `log/universal-ir`. 
You can print your log at time by running `tail -f log/universal-ir/train_universal-ir_***.log -n 100`.

### Evaluation
To evalute our method on image restoration, please modify the benchmark path and model path and run

```bash
cd universal-image-restoration/config/universal-ir
python test.py -opt=options/test.yml
```

### Gradio
Here we provide an [app.py](https://github.com/Algolzw/daclip-uir/tree/main/universal-image-restoration/config/daclip-sde/app.py) file for testing your own images. Before that, you need to download the pretrained weights ([DA-CLIP](https://drive.google.com/file/d/1A6u4CaVrcpcZckGUNzEXqMF8x_JXsZdX/view?usp=sharing) and [UIR](https://drive.google.com/file/d/1eXsyrmAbWOvhIY4Wbt5v4IxaggA5aZMG/view?usp=sharing)) and modify the model path in `options/test.yml`. Then by simply running `python app.py`, you can open `http://localhost:7860` to test the model. (We also provide several images with different degradations in the `images` dir).


### Results

![daclip](figs/UIR_results_radar.jpg)

<details>
<summary><strong>Unified Image Restoration</strong> (click to expand) </summary>

![daclip](figs/results-UIR.jpg)

</details>

<details>
<summary><strong>Degradation-Specific Restoration</strong> (click to expand) </summary>

![daclip](figs/results_single.jpg)

</details>



---

**Acknowledgment:** Our DA-CLIP is based on [IR-SDE](https://github.com/Algolzw/image-restoration-sde) and [open_clip](https://github.com/mlfoundations/open_clip). Thanks for their code!

#### Contact
If you have any question, please contact: ziwei.luo@it.uu.se


### Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```
@article{luo2023controlling,
  title={Controlling Vision-Language Models for Universal Image Restoration},
  author={Luo, Ziwei and Gustafsson, Fredrik K and Zhao, Zheng and Sj{\"o}lund, Jens and Sch{\"o}n, Thomas B},
  journal={arXiv preprint arXiv:2310.01018},
  year={2023}
}
```

---


#### --- Thanks for your interest! --- ####

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Algolzw/daclip-uir)

</details>

