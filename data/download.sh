#!/usr/bin/env bash

# download caption_datasets
axel -n 100 http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
mkdir caption_datasets
mv *.json caption_datasets

# download pretrain model
axel -n 100 https://s3.amazonaws.com/resnext/imagenet_models/resnext_101_32x4d.t7
axel -n 100 https://s3.amazonaws.com/resnext/imagenet_models/resnext_101_64x4d.t7
axel -n 100 https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7
axel -n 100 https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth
