from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import numpy as np
import random
from torchvision import transforms as trn
from torch.autograd import Variable
import torch


from multiprocessing.dummy import Process, Queue, Pool

preprocess = trn.Compose([
    # trn.ToPILImage(),
    # trn.Scale(256),
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DataLoader():

    def reset_iterator(self, split):
        self._prefetch_process[split].terminate()
        self._prefetch_process[split].join()
        self._prefetch_process[split] = BlobFetcher(split, self)
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
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

        print('assigned %d images to split train'%len(self.split_ix['train']))
        print('assigned %d images to split val'%len(self.split_ix['val']))
        print('assigned %d images to split test'%len(self.split_ix['test']))

        self.iterators = {'train':0, 'val':0 ,'test':0}

        self._prefetch_process = {}
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self)

        def cleanup():
            for split in self.iterators.keys():
                self._prefetch_process[split].terminate()
                self._prefetch_process[split].join()

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img


        image_batch = np.zeros([batch_size]+list(self.input_h5['images'].shape[1:]),dtype='float32')
        label_batch = np.zeros([batch_size*seq_per_img,self.seq_length+2],dtype='int')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):

            image_one_batch, ix, tmp_wrapped = self._prefetch_process[split].get()

            image_batch[i:(i + 1), :, :, :] = image_one_batch

            ix1 = self.label_start_ix[ix] - 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1

            assert ncap > 0,'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < seq_per_img:
                seq = np.zeros([seq_per_img,self.seq_length],dtype='int')
                for q in range(seq_per_img):
                    ix1 = random.randint(ix1,ix2)
                    seq[q,:] = self.input_h5['labels'][ix1, :self.seq_length]
            else:
                ix1 = random.randint(ix1,ix2-seq_per_img+1)
                seq = self.input_h5['labels'][ix1:ix1+seq_per_img,:self.seq_length]

            label_batch[i*seq_per_img:(i+1)*seq_per_img,1:self.seq_length+1] = seq

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.input_h5['labels'][self.label_start_ix[ix]-1: self.label_end_ix[ix]-1])

            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        data = {}

        data['images'] = image_batch
        data['labels'] = label_batch
        data['gts'] = gts
        data['bounds'] = {'it_pos_now':self.iterators[split], 'it_max':len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data


class BlobFetcher():

    def __init__(self, split, dataloader):

        self.split = split
        self.dataloader = dataloader

        self.pool = Pool(4)
        self.fifo = []

    def reset(self):

        if len(self.fifo) == 0:
            self.cur_idx = self.dataloader.iterators[self.split]
        split_ix = self.dataloader.split_ix[self.split]
        for i in xrange(512 - len(self.fifo)):
            ix = split_ix[self.cur_idx]
            if self.cur_idx + 1 >= len(split_ix):
                self.cur_idx = 0
            else:
                self.cur_idx += 1
            self.fifo.append(self.pool.apply_async(self._get_minibatch, (ix,)))

    def terminate(self):
        self.pool.terminate()

    def join(self):
        self.pool.join()

    def _get_next_minibatch_inds(self):
        split_ix = self.dataloader.split_ix[self.split]
        max_index = len(split_ix)
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next
        ix = split_ix[ri]

        return ix, wrapped

    def _get_minibatch(self, ix):
        wrapped = False
        if ix == self.dataloader.split_ix[self.split][-1]:
            wrapped = True
        image = self.dataloader.input_h5['images'][ix]
        image = image.astype('float32') / 255.0

        # I = image.astype('float32') / 255.0
        # I = torch.from_numpy(I).cuda()
        # I = Variable(preprocess(I), volatile=True)

        return (image, ix, wrapped)

    def get(self):
        if len(self.fifo) < 400:
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.fifo.pop(0).get()

        assert tmp[1] == ix, "ix not equal"
        assert tmp[2] == wrapped, "wrapped not equal"

        return tmp
