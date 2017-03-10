#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

## train densenet-169 
python -u train_densenet.py --data-dir data/imagenet --data-type imagenet --depth 169 --batch-size 256 --growth-rate 32 --drop-out 0 --reduction 0.5 --gpus=6,7,8,9