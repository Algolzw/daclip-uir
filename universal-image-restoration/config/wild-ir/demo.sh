#!/bin/bash

#############################################################
### training ###

# for single GPU
python train.py -opt=options/train.yml

# for multiple GPUs
# torchrun --nproc_per_node 2 -m train -opt=options/train.yml

#############################################################

### testing ###
# python test.py -opt=options/test.yml
# python inference.py -opt=options/inference.yml

#############################################################
