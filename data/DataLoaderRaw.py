from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
from scipy.misc import imread, imresize
import db.lmdb_data as lmdb_data
import io

class DataLoaderRaw():

    def __init__(self, opt):
        self.opt = opt
        self.folder_path = opt.get('folder_path','')
        self.batch_size = opt.get('batch_size',1)
        self.start = opt.get('start', 0)
        self.num = opt.get('num', 7500)
        self.seq_per_img = 1

        self.use_bu_att = opt.get('use_bu_att', False)
        self.input_bu = opt.get('input_bu', False)
        self.bu_size = opt.get('bu_size', False)
        self.bu_feat_size = opt.get('bu_feat_size', False)

        if self.use_bu_att:
            self.lmdb_bu = lmdb_data.lmdb_data(self.input_bu)
            self.lmdb_bu.open_for_read()

        self.files = []
        self.ids = []

        def isImage(f):
            supportExt = ['.jpg','.JPG','.JPEG','.png','.PNG','.ppm','.PPM']
            for ext in supportExt:
                start_idx = f.rfind(ext)
                if start_idx >= 0 and start_idx + len(ext) == len(f):
                    return True
            return False

        print('start: ',self.start,'num: ',self.num)
        n = 0
        for root, dirs, files in os.walk(self.folder_path, topdown=False):
            for file in files:
                fullpath = os.path.join(self.folder_path, file)
                if isImage(fullpath):
                    if self.num == -1 or (n >= self.start and n < self.start + self.num):
                        self.files.append(fullpath)
                        self.ids.append(file.split('.')[0])
                    n = n + 1

        self.N = len(self.files)

        print('DataLoaderRaw found', self.N, ' images')

        self.iterator = 0


    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        image_size = 256
        img_batch = np.ndarray([batch_size, 3, image_size, image_size], dtype='float32')
        max_index = self.N
        wrapped = False
        infos = []

        if self.use_bu_att:
            bu_batch = np.ndarray([batch_size, self.bu_size, self.bu_feat_size], dtype='float32')


        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterator = ri_next

            img = imread(self.files[ri])

            img = imresize(img, (image_size, image_size))

            if len(img.shape) == 2:
                img = img[:,:,np.newaxis]
                img = np.concatenate((img,img,img),axis=2)

            img = img.transpose(2, 0, 1)

            img_batch[i] = img

            info_struct = {}
            info_struct['id'] = self.ids[ri]
            info_struct['file_path'] = self.files[ri]
            infos.append(info_struct)

            if self.use_bu_att:
                # print(str(info_struct['id']) + ".jpg")
                value = self.lmdb_bu.get(str(info_struct['id']) + ".jpg")
                data = np.load(io.BytesIO(value))
                data_x = data['x'].tolist()
                bu_batch[i] = data_x['features']

        data = {}
        data['images'] = img_batch
        data['bounds'] = {'it_pos_now':self.iterator, 'it_max':self.N, 'wrapped':wrapped}
        data['infos'] = infos

        if self.use_bu_att:
            data['bus'] = bu_batch

        return data

    def reset_iterator(self, split):
        self.iterator = 0

    def get_vocab_size(self):
        return len(self.it_to_word)

    def get_vocab(self):
        return self.ix_to_word