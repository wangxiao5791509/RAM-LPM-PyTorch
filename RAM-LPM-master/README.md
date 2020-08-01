# Recurrent Attention Model with Log-Polar Mapping (RAM-LPM) 

## Abstract

In image analysis, convolution is usually applied to a rectangular field of view (FOV) to extract and learn features. We developed a convolutional neural network (CNN) that processes data from a log-polar FOV. Information is densely sampled near the center of the FOV, similar to the fovea of mammals, whereas in the periphery, the sampling resolution is lower. This design balances the high resolution processing near the center of the FOV, and low resolution data sampling over large area. Our FOV can be rotation invariant. We also demonstrate the robustness of our network against adversarial attacks. This repo include the code that reproduces the results in [tbd]. 

# Setup the environment for experiments

## Install dependencies with conda

```shell
$ wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh
$ bash ~/miniconda.sh -b -p $HOME/miniconda
$ export PATH="$HOME/miniconda/bin:$PATH"
$ conda update conda
$ conda env create -f environment.yml
$ conda activate ram_lpm
$ pip install -e .
```

## Installation with Docker [WIP]

```shell
$ docker build -t kiritani_ono:1.0 .
$ docker container run -id --rm kiritani_ono:1.0 --name ex1
```

# Experiments

## 1. Training on ImageNet

Download ImageNet dataset and put the training dataset in `./train`, and the validation dataset in `./val`.

Then, train the RAM-LPM:

```shell
$ python scripts/run3.py
```
.

Run the inference on the validation data:

```shell
$ python scripts/inference.py
```
.

This generates a figure summarizing the results: `imagenet_training/summary.png`.

## 2. Evaluation of Robustness against SPSA and PGD attacks

You need to specify which attack is used, how images are preprocessed (image size), which images are modified (the panda or imagenet validation), and saved checkpoint.

```shell
$ python scripts/adversarial_attacks.py spsa 224 imagenet imagenet_training/best_model.ckpt
$ python scripts/adversarial_attacks.py spsa original imagene timagenet_training/best_model.ckpt
$ python scripts/adversarial_attacks.py pgd 224 imagenet imagenet_training/best_model.ckpt
$ python scripts/adversarial_attacks.py pgd original imagenet imagenet_training/best_model.ckpt
```

Results are logged in `./adversarial_exp/`.

## 3. Performance of RAM-LPM on SIM2MNIST

```shell
$ python scripts/run2.py with H=20 W=24 batch_size=256 std=0.16 epochs=500 model_name=ramlpm init_lr=0.001 init_lr_where=0.00001 work_dir=sim2exp num_workers=4 upsampling_factor_r=1 upsampling_factor_theta=1 dataset=sim2mnist "kernel_sizes_conv2d=[[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]" "kernel_sizes_pool=[[1,1],[1,1],[3,3],[1,1],[1,1],[1,1],[2,8]]" r_max=0.8 r_min=0.05 "kernel_dims=[1,32,32,64,64,64,64,64]" "kernel_sizes_conv2d_where=[[3,3],[3,3],[3,3],[3,3]]" "kernel_sizes_pool_where=[[3,3],[1,1],[1,1],[2,3]]" "kernel_dims_where=[1,32,32,32,4]" num_glimpses=20
```

## References

* `Recurrent Models of Visual Attention`
  * https://arxiv.org/abs/1406.6247

* `Rotation Equivariance and Invariance in Convolutional Neural Networks`
  * https://arxiv.org/pdf/1805.12301.pdf

* `Learning Rotation-Invariant and Fisher Discriminative Convolutional Neural Networks for Object Detection`

# Acknowledgement

Large part of the code is imported from `https://github.com/kevinzakka/recurrent-visual-attention`.
