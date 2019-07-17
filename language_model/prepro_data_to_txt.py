#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import jieba
import os
import json
import argparse
from random import shuffle, seed
import string
import h5py
import numpy as np
from scipy.misc import imread, imresize
import time
from tqdm import tqdm
import hashlib

def build_sentences(imgs_train, imgs_val, output_train, output_val):

    len_imgs_train = len(imgs_train)
    len_imgs_val = len(imgs_val)
    len_imgs = len_imgs_train + len_imgs_val

    file_train = open(output_train, 'w')
    file_val = open(output_val, 'w')

    len_split = len_imgs_train + 25000
    for i in tqdm(range(len_imgs)):
        if i < len_imgs_train:
            img = imgs_train[i]
        else:
            img = imgs_val[i-len_imgs_train]

        if i < len_split:
            split = 'train'
        else:
            split = 'val'

        captions = img['caption']

        for caption in captions:
            if split == 'train':
                file_train.write(caption + "\n")
            elif split == 'val':
                file_val.write(caption + "\n")

    file_train.close()
    file_val.close()


def main(params):

    imgs_train = json.load(open(params['input_json_train'], 'r'))
    imgs_val = json.load(open(params['input_json_val'], 'r'))

    output_train = params['output_train']
    output_val = params['output_val']

    build_sentences(imgs_train, imgs_val, output_train, output_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    # http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    parser.add_argument('--root_path', default='/home/amds/caption/dataset/aic', help='input json file to process into hdf5')
    parser.add_argument('--input_json_train', default='ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', help='input json file to process into hdf5')
    parser.add_argument('--input_json_val', default='ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', help='input json file to process into hdf5')

    parser.add_argument('--output_train', default='aic_train.txt', help='input json file to process into hdf5')
    parser.add_argument('--output_val', default='aic_val.txt', help='input json file to process into hdf5')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    params['input_json_train'] = os.path.join(params['root_path'], params['input_json_train'])
    params['input_json_val'] = os.path.join(params['root_path'], params['input_json_val'])

    params['output_train'] = os.path.join(params['root_path'], params['output_train'])
    params['output_val'] = os.path.join(params['root_path'], params['output_val'])

    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)

