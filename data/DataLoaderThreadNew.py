from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import numpy as np
import random
import torch
from torchvision import transforms as trn
import threading
import Queue
from torch.autograd import Variable
import misc.utils as utils
import time
from scipy.misc import imread, imresize
import vis.visual_conv as vis_conv
import os
import models
import db.lmdb_data as lmdb_data
import io
import data.img_aug_utils as img_aug_utils
import memcache

# preprocess = trn.Compose([
#     trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
# )

# preprocess_img = trn.Compose([
#     trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
# )



class DataLoaderThreadNew():

    def reset_iterator(self, split):
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, is_only_test = False):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.images_root = opt.images_root
        self.image_size = opt.image_size
        self.img_padding_max = opt.img_padding_max

        if opt.use_pre_feat:
            self.lmdb = lmdb_data.lmdb_data(opt.path_lmdb)
            self.lmdb.open_for_read()

        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is', self.vocab_size)

        print('DataLoader laoding h5 file: ',opt.input_h5)
        self.input_h5 = h5py.File(self.opt.input_h5, 'r')

        self.use_lmdb_img = getattr(opt, 'use_lmdb_img', False)
        if self.use_lmdb_img:
            self.lmdb_img = lmdb_data.lmdb_data(opt.path_lmdb_img)
            self.lmdb_img.open_for_read()

        self.use_heavy_aug = getattr(opt, 'use_heavy_aug', False)
        self.use_memory_img = getattr(opt, 'use_memory_img', False)
        if self.use_memory_img:
            self.memory_img = {}

        self.use_memcache_img = getattr(opt, 'use_memcache_img', False)
        if self.use_memcache_img:
            self.memcache_client = memcache.Client([opt.memcache_host])

        # if models.is_inception(opt.cnn_model):
        #     img_mean = [0.5, 0.5, 0.5]
        #     img_std = [0.5, 0.5, 0.5]
        # else:
        #     img_mean = [0.485, 0.456, 0.406]
        #     img_std = [0.229, 0.224, 0.225]
        #
        # vgg_mean = torch.FloatTensor(img_mean).cuda().view(1, 3, 1, 1)
        # self.img_mean = vgg_mean.expand(self.batch_size, 3, self.image_size, self.image_size)
        #
        # vgg_std = torch.FloatTensor(img_std).cuda().view(1, 3, 1, 1)
        # self.img_std = vgg_std.expand(self.batch_size, 3, self.image_size, self.image_size)

        if models.is_inception(opt.cnn_model):
            self.preprocess = trn.Compose([trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.preprocess = trn.Compose([trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


        # image_size = self.input_h5['images'].shape
        # self.num_images = image_size[0]
        # print('image size', image_size)
        # print('read %d image'%(self.num_images))

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
        # random.shuffle(self.split_ix['val'])
        # random.shuffle(self.split_ix['test'])

        print('assigned %d images to split train'%len(self.split_ix['train']))
        print('assigned %d images to split val'%len(self.split_ix['val']))
        print('assigned %d images to split test'%len(self.split_ix['test']))

        self.iterators = {'train':0, 'val':0 ,'test':0}

        self.queues = {}
        self.flags = {}
        self.threads = {}

        if not is_only_test:

            for split in ['train']:
                self.queues[split] = Queue.Queue(maxsize=1)
                self.flags[split] = True
                thread = threading.Thread(target=get_batch_worker_thread, args=(split, self, opt.batch_size))
                thread.setDaemon(True)
                thread.start()
                self.threads[split] = thread
                # have a bug, when load in the same time
                # so we sleep 1s for old data load
                # time.sleep(1)

        # import atexit
        # atexit.register(terminate)

    def terminate(self):
        for split in self.split_ix.keys():
            self.flags[split] = False
            # self.threads[split].join()

    def get_batch(self, split, batch_size):
        if split == 'train':
            return self.queues[split].get()
        else:
            return get_batch_worker(split, self, batch_size)
        # return self.queues[split].get()


def get_batch_worker_thread(split, loader, batch_size):
    while loader.flags[split]:
        data = get_batch_worker(split, loader, batch_size)
        loader.queues[split].put(data)


def get_batch_worker(split, loader, batch_size):

    # batch_size = loader.batch_size
    seq_per_img = loader.seq_per_img
    split_ix = loader.split_ix[split]

    # image_size = 256
    # image_size = 224

    if loader.opt.use_pre_feat:
        img_batch = torch.FloatTensor(batch_size, loader.opt.att_feat_size, loader.opt.pool_size, loader.opt.pool_size)
    else:
        img_batch = torch.FloatTensor(batch_size, 3, loader.image_size, loader.image_size)

    label_batch = np.zeros([batch_size * seq_per_img, loader.seq_length+1], dtype='int')
    mask_batch = np.zeros([batch_size * seq_per_img, loader.seq_length+1], dtype='float32')
    token_batch = np.zeros([batch_size * seq_per_img, loader.vocab_size+1], dtype='int')

    max_index = len(split_ix)
    wrapped = False

    infos = []
    gts = []

    # start = time.time()
    for i in range(batch_size):

        ri = loader.iterators[split]
        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            wrapped = True
        loader.iterators[split] = ri_next
        ix = split_ix[ri]

        img_info = loader.info['images'][ix]

        if loader.opt.use_pre_feat:
            value = loader.lmdb.get(str(img_info['id']))
            data = np.load(io.BytesIO(value))
            Ir = data['x']
            img = torch.from_numpy(Ir)
        else:
            # fetch image
            if loader.use_lmdb_img:
                value = loader.lmdb_img.get(str(img_info['file_path']))
                data = np.load(io.BytesIO(value))
                I = data['img']
            elif loader.use_memory_img:
                img_key = str(img_info['file_path'])
                if loader.memory_img.has_key(img_key):
                    I = loader.memory_img[img_key]
                else:
                    I = imread(os.path.join(loader.images_root, img_info['file_path']))
                    I = imresize(I, (loader.image_size, loader.image_size))
                    loader.memory_img[img_key] = I
            elif loader.use_memcache_img:
                img_key = str(img_info['file_path'])
                value = loader.memcache_client.get(img_key)
                if value is None:
                    I = imread(os.path.join(loader.images_root, img_info['file_path']))
                    I = imresize(I, (loader.image_size, loader.image_size))
                    loader.memcache_client.set(img_key, I)
                else:
                    data = np.load(io.BytesIO(value))
                    I = data['img']
            else:
                I = imread(os.path.join(loader.images_root, img_info['file_path']))
            # handle grayscale input images
            if len(I.shape) == 2:
                I = I[:, :, np.newaxis]
                I = np.concatenate((I, I, I), axis=2)

            try:
                if split == 'train':

                    span_width = random.randint(0, loader.img_padding_max)
                    Ir = imresize(I, (loader.image_size + span_width, loader.image_size + span_width))
                    rx = random.randint(0, span_width)
                    ry = random.randint(0, span_width)
                    Ir = Ir[rx: loader.image_size + rx, ry: loader.image_size + ry, :]

                    if loader.use_heavy_aug:
                        Ir = img_aug_utils.img_aug_normal(Ir)

                else:
                    Ir = imresize(I, (loader.image_size, loader.image_size))
            except:
                print('failed resizing image %s - see http://git.io/vBIE0' % (img_info['file_path']))
                raise

            # and swap order of axes from (256,256,3) to (3,256,256)
            Ir = Ir.astype('float32') / 255.0
            Ir = Ir.transpose(2, 0, 1)

            if split == 'train':
                # random flip
                # vis_conv.show_img(img.transpose(1,2,0))
                if loader.opt.use_mirror and random.randint(0, 99) >= 50:
                    Ir = np.flip(Ir, 2).copy()
                # vis_conv.show_img(img.transpose(1,2,0))

            img = torch.from_numpy(Ir)
            img = loader.preprocess(img)

        img_batch[i] = img

        ix1 = loader.label_start_ix[ix] - 1
        ix2 = loader.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1
        assert ncap > 0,'an image does not have any label. this can be handled but right now isn\'t'

        token = np.zeros([seq_per_img, loader.vocab_size + 1], dtype='int')
        if ncap < seq_per_img:
            seq = np.zeros([seq_per_img, loader.seq_length],dtype='int')
            for q in range(seq_per_img):
                ix1 = random.randint(ix1, ix2)
                seq[q,:] = loader.input_h5['labels'][ix1, :loader.seq_length]
        else:
            ix1 = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = loader.input_h5['labels'][ix1: ix1 + seq_per_img, :loader.seq_length]

        for k in range(seq_per_img):
            token[k, seq[k]] = 1

        label_batch[i*seq_per_img: (i+1)*seq_per_img, :loader.seq_length] = seq
        token_batch[i*seq_per_img: (i+1)*seq_per_img] = token

        # Used for reward evaluation
        gts.append(loader.input_h5['labels'][loader.label_start_ix[ix] - 1: loader.label_end_ix[ix]])

        info_dict = {}
        info_dict['id'] = loader.info['images'][ix]['id']
        info_dict['file_path'] = loader.info['images'][ix]['file_path']
        if 'image_id' in loader.info['images'][ix].keys():
            info_dict['image_id'] = loader.info['images'][ix]['image_id']
        infos.append(info_dict)
    # print(time.time() - start)
    nonzeros = np.array(map(lambda x: (x != 0).sum()+1, label_batch))

    if loader.opt.loss_weight_type == 0:
        weight = np.linspace(loader.opt.loss_weight_start, loader.opt.loss_weight_stop, loader.seq_length+1)
        for ix, row in enumerate(mask_batch):
            mask_len = nonzeros[ix]
            row[:mask_len] = weight[:mask_len]
    elif loader.opt.loss_weight_type == 1:
        half_len = loader.seq_length//2
        weight = np.linspace(loader.opt.loss_weight_stop, loader.opt.loss_weight_start, half_len)
        weight1 = np.linspace(loader.opt.loss_weight_start, loader.opt.loss_weight_stop, half_len+1)
        for ix, row in enumerate(mask_batch):
            mask_len = nonzeros[ix]
            if mask_len <= half_len:
                row[:mask_len] = weight[:mask_len]
            else:
                row[:half_len] = weight[:half_len]
                row[half_len:mask_len] = weight1[:mask_len-half_len]
    else:
        for ix, row in enumerate(mask_batch):
            mask_len = nonzeros[ix]
            row[:mask_len] = 1


    if wrapped and split == 'train':
        random.shuffle(loader.split_ix[split])

    data = {}

    # images = torch.from_numpy(img_batch)
    # if split == 'train':
    #     data_augment = True
    # else:
    #     data_augment = False
    #
    #

    # if loader.opt.data_norm:
    #     images = utils.prepro_norm(images, data_augment)
    # else:
    #     images = utils.prepro(images, data_augment)

    # for inception
    # images.add_(loader.img_mean)
    # images.div_(loader.img_std)

    images = Variable(img_batch.cuda(), requires_grad=False)

    labels = torch.from_numpy(label_batch).cuda()
    labels = Variable(labels, requires_grad=False)

    targets = torch.cat([labels.data.new(labels.size(0), 1).fill_(0), labels.data[:, :-1]], 1)
    targets = Variable(targets, requires_grad=False)

    masks = torch.from_numpy(mask_batch).cuda()
    masks = Variable(masks, requires_grad=False)

    tokens = torch.from_numpy(token_batch).cuda().float()
    tokens = Variable(tokens, requires_grad=False)

    data['images'] = images
    data['labels'] = labels
    data['targets'] = targets
    data['masks'] = masks
    data['tokens'] = tokens
    data['gts'] = gts
    data['bounds'] = {'it_pos_now':loader.iterators[split], 'it_max':len(loader.split_ix[split]), 'wrapped': wrapped}
    data['infos'] = infos

    return data