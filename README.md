# A MXNet implementation of DenseNet with BC structure

This a [MXNet](http://mxnet.io/) implementation of DenseNet-BC architecture as described in the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v3.pdf) by Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten.

![](title.png)

This implementation only focus on imagenet'12 dataset at present. The training procedure is ongoing. So, I hope anyone who are mxnet fun can test this code with me. When I finish, I will update more information about training and validation.

Their official implementation and many other third-party implementations can be found in the [liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet) repo on GitHub.



This is a basic dense block (figure is modified from the [original paper](https://arxiv.org/pdf/1608.06993v3.pdf)). Each layer takes all preceding feature maps as input. It is a very interesting and simple design.

![](dense-block.png)

I very thanks [tornadomeet, Wei Wu](https://github.com/tornadomeet). This implementation of DenseNet is adapted from his [Resnet](https://github.com/tornadomeet/ResNet) codes. I also refered other third-part implementations, such as 
two version of PyTorch

1. [Brandon Amos](https://github.com/bamos)
(https://github.com/bamos/densenet.pytorch/blob/master/densenet.py)  
2. [Andreas Veit](https://github.com/andreasveit)
(https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py)

one version of MXNet (without BC structure)

3. [Nicatio](https://github.com/Nicatio)
(https://github.com/Nicatio/Densenet/blob/master/mxnet/symbol_densenet.py)

#Requirements

Install MXNet on a mechine with CUDA GPU, and it's better also installed with [cuDNN v5](https://developer.nvidia.com/cudnn)
Please fix the randomness if you want to train your own model and using [Wei Wu](https://github.com/dmlc/mxnet/pull/3001/files) solution.

#Data

ImageNet'12 dataset
Imagenet 1000 class dataset with 1.2 million images. Because this dataset is about 120GB, so you have to download by yourself. Sorry for this inconvenience.

#How to Train

For this part, before you want to train your model, please read the suggestion from [Wei Wu](https://github.com/tornadomeet/ResNet) first. In his page, there is a very detailed information about how to prepare your data. 

When you finised data preparation, please make sure the data locates the same folder of source codes. then you can
run the training cmd just like this (here, I use 4 gpus for training):

python -u train_densenet.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 256 --gpus=6,7,8,9

Maybe you should change batch-size from 256 to 128 due to the memory size of GPU.



