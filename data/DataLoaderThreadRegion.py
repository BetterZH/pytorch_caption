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

# preprocess = trn.Compose([
#     trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
# )

# preprocess_img = trn.Compose([
#     trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
# )



class DataLoaderThreadRegion():

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

        self.att_size = opt.att_size

        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is', self.vocab_size)

        print('DataLoader laoding h5 file: ',opt.input_h5)
        self.input_h5 = h5py.File(self.opt.input_h5, 'r')

        print('DataLoader laoding bu file: ',opt.input_bu)
        self.lmdb_bu = lmdb_data.lmdb_data(opt.input_bu)
        self.lmdb_bu.open_for_read()

        if models.is_inception(opt.cnn_model):
            self.preprocess = trn.Compose([trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.preprocess = trn.Compose([trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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

def image_aug(split, loader, I):
    if split == 'train':
        span_width = random.randint(0, loader.img_padding_max)
        Ir = imresize(I, (loader.image_size + span_width, loader.image_size + span_width))
        rx = random.randint(0, span_width)
        ry = random.randint(0, span_width)
        Ir = Ir[rx: loader.image_size + rx, ry: loader.image_size + ry, :]
    else:
        Ir = imresize(I, (loader.image_size, loader.image_size))
    return Ir

def get_batch_worker(split, loader, batch_size):

    # batch_size = loader.batch_size
    seq_per_img = loader.seq_per_img
    split_ix = loader.split_ix[split]

    img_batch = torch.FloatTensor(batch_size * (1+loader.att_size), 3, loader.image_size, loader.image_size)

    label_batch = np.zeros([batch_size * seq_per_img, loader.seq_length+1], dtype='int')
    mask_batch = np.zeros([batch_size * seq_per_img, loader.seq_length+1], dtype='float32')
    token_batch = np.zeros([batch_size * seq_per_img, loader.vocab_size+1], dtype='int')

    max_index = len(split_ix)
    wrapped = False

    infos = []
    gts = []

    # start = time.time()
    k = 0
    for i in range(batch_size):

        ri = loader.iterators[split]
        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            wrapped = True
        loader.iterators[split] = ri_next
        ix = split_ix[ri]

        img_info = loader.info['images'][ix]

        if img_info.has_key("image_id"):
            image_id = str(img_info['image_id']) + ".jpg"
        else:
            image_id = str(img_info["id"])
        value = loader.lmdb_bu.get(image_id)
        data = np.load(io.BytesIO(value))
        data_x = data['x'].tolist()
        boxes = data_x['boxes']

        # fetch image
        I = imread(os.path.join(loader.images_root, img_info['file_path']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        for j in range(loader.att_size+1):

            if j == 0:
                img = I
            else:
                box = boxes[j-1]
                x_1 = int(box[0])
                y_1 = int(box[1])
                x_2 = int(box[2])
                y_2 = int(box[3])
                img = I[y_1:y_2, x_1:x_2]

            try:
                Ir = image_aug(split, loader, img)
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

            img_batch[k] = img
            k += 1

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


    if wrapped:
        random.shuffle(loader.split_ix[split])

    data = {}

    images = Variable(img_batch.cuda(), requires_grad=False)

    labels = torch.from_numpy(label_batch).cuda()
    labels = Variable(labels, requires_grad=False)

    masks = torch.from_numpy(mask_batch).cuda()
    masks = Variable(masks, requires_grad=False)

    tokens = torch.from_numpy(token_batch).cuda().float()
    tokens = Variable(tokens, requires_grad=False)

    data['images'] = images
    data['labels'] = labels
    data['masks'] = masks
    data['tokens'] = tokens
    data['gts'] = gts
    data['bounds'] = {'it_pos_now':loader.iterators[split], 'it_max':len(loader.split_ix[split]), 'wrapped': wrapped}
    data['infos'] = infos

    return data