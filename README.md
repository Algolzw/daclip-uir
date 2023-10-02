
## Controlling Vision-Language Models for Universal Image Restoration <br><sub>Official PyTorch Implementations of [[DA-CLIP]](). </sub>

[[Project Page](https://algolzw.github.io/daclip-uir/index.html)] | [[Paper]()]

![daclip](figs/teaser.jpg)


## How to Run the Code?

### Install

We advise you first create a virtual environment with:

```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt

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

```

Then get into the `codes/config/daclip-sde` directory and modify the dataset paths in option files in `options/train.yml` and `options/tes.yml`. 

You can add more tasks or datasets to both `train` and `val` directories and add the degradation word to `distortion`.


### Train
The main code for training is in `codes/config/daclip-sde` and the core network for DA-CLIP is in `codes/open_clip/daclip_model.py`.

* Put the pretrained [**DA-CLIP weights**](https://drive.google.com/file/d/1A6u4CaVrcpcZckGUNzEXqMF8x_JXsZdX/view?usp=sharing) to `pretrained` directory and check the `daclip` path.

* You can then train the model following below bash scripts:

```bash
cd codes/config/daclip-sde

# For single GPU:
python3 train.py -opt=options/train.yml

# For distributed training, need to change the gpu_ids in option file
python3 -m torch.distributed.launch --nproc_per_node=2 --master_poer=4321 train.py -opt=options/train.yml --launcher pytorch
```

The models and training logs will save in `log/universal-ir`. 
You can print your log at time by running `tail -f log/universal-ir/train_universal-ir_***.log -n 100`.

### Evaluation
To evalute our method, please modify the benchmark path and model path and run

```bash
cd codes/config/universal-ir
python test.py -opt=options/test.yml
```

### Results

<details>
<summary><strong>Degradation-Specific Restoration</strong> (click to expand) </summary>
![daclip](figs/results_single.jpg)
</details>

<details>
<summary><strong>Unified Image Restoration</strong> (click to expand) </summary>
![daclip](figs/results-UIR.jpg)
</details>

<details>
<summary><strong>Radar Results</strong> (click to expand) </summary>
![daclip](figs/UIR_results_radar.jpg)
</details>

---

**Acknowledgment:** Our DA-CLIP is based on [IR-SDE](https://github.com/Algolzw/image-restoration-sde) and [open_clip](https://github.com/mlfoundations/open_clip). Thanks for their code!

#### Contact
If you have any question, please contact: ziwei.luo@it.uu.se



#### --- Thanks for your interest! --- ####
<!--![visitors](https://visitor-badge.laobi.icu/badge?page_id=Algolzw/daclip-uir)-->
