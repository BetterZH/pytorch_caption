from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import numpy as np
import random
import torch
from torchvision import transforms as trn
from torch.autograd import Variable
import misc.utils as utils

preprocess = trn.Compose([
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class DataLoader():

    def reset_iterator(self, split):
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img

        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is', self.vocab_size)

        print('DataLoader laoding h5 file: ',opt.input_h5)
        self.input_h5 = h5py.File(self.opt.input_h5, 'r')

        image_size = self.input_h5['images'].shape
        self.num_images = image_size[0]
        print('image size', image_size)
        print('read %d image'%(self.num_images))

        # load in the sequence data
        seq_size = self.input_h5['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        self.label_start_ix = self.input_h5['label_start_ix'][:]
        self.label_end_ix = self.input_h5['label_end_ix'][:]

        # separate out indexes for each of the provided splits
        self.split_ix = {'train':[],'val':[],'test':[]}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:
                self.split_ix['train'].append(ix)

        random.shuffle(self.split_ix['train'])
        random.shuffle(self.split_ix['val'])
        random.shuffle(self.split_ix['test'])

        print('assigned %d images to split train'%len(self.split_ix['train']))
        print('assigned %d images to split val'%len(self.split_ix['val']))
        print('assigned %d images to split test'%len(self.split_ix['test']))

        self.iterators = {'train':0, 'val':0 ,'test':0}

    def get_batch(self, split):

        split_ix = self.split_ix[split]
        batch_size = self.batch_size
        seq_per_img = self.seq_per_img
        image_size = 256

        img_batch = np.zeros([batch_size, 3, image_size, image_size],dtype='float32')
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length+1], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length+1], dtype='float32')

        max_index = len(split_ix)
        wrapped = False

        infos = []

        for i in range(batch_size):

            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = split_ix[ri]

            # fetch image
            img = self.input_h5['images'][ix, :, :, :]
            img_batch[i] = img

            ix1 = self.label_start_ix[ix] - 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1
            assert ncap > 0,'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < seq_per_img:
                seq = np.zeros([seq_per_img, self.seq_length],dtype='int')
                for q in range(seq_per_img):
                    ix1 = random.randint(ix1, ix2)
                    seq[q,:] = self.input_h5['labels'][ix1, :self.seq_length]
            else:
                ix1 = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.input_h5['labels'][ix1: ix1 + seq_per_img, :self.seq_length]

            label_batch[i*seq_per_img: (i+1)*seq_per_img, :self.seq_length] = seq

            info_dict = {}
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        nonzeros = np.array(map(lambda x: (x != 0).sum()+1, label_batch))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        if wrapped:
            random.shuffle(self.split_ix[split])

        images = torch.from_numpy(img_batch).cuda()
        images = utils.prepro(images, True)
        images = Variable(images, requires_grad=False)

        labels = torch.from_numpy(label_batch).cuda()
        labels = Variable(labels, requires_grad=False)

        masks = torch.from_numpy(mask_batch).cuda()
        masks = Variable(masks, requires_grad=False)

        data = {}
        data['images'] = images
        data['labels'] = labels
        data['masks'] = masks
        data['bounds'] = {'it_pos_now':self.iterators[split], 'it_max':len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data
