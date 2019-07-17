import os
import json
import argparse
from random import shuffle, seed
import string
import h5py
import numpy as np
from scipy.misc import imread, imresize
import time
from db import lmdb_data
from scipy.misc import imread, imresize
import io
import torch.nn as nn
import models
import torch
from torchvision import transforms as trn
import random
from torch.autograd import Variable
import math
import Queue
import threading
import copy

flag = True
queue_cnn = Queue.Queue(maxsize=3)
queue_fc = Queue.Queue(maxsize=3)

def thread_cnn_worker(model_cnn):

    while True:
        data = queue_cnn.get()
        is_end = data['is_end']
        if is_end == 1:
            data_fc = {}
            data_fc['is_end'] = is_end
            queue_fc.put(data_fc)
            break

        img_batch = data['img_batch']
        batch_num = data['batch_num']
        ids = data['ids']

        images = Variable(img_batch)
        fc_feats = model_cnn(images)

        data_fc = {}
        data_fc['batch_num'] = batch_num
        data_fc['ids'] = copy.deepcopy(ids)
        data_fc['fc_feats'] = fc_feats.data.clone()
        data_fc['is_end'] = is_end
        queue_fc.put(data_fc)


    print('cnn end')

def thread_lmdb_worker(output_lmdb):

    lmdb = lmdb_data.lmdb_data(output_lmdb, int(1e11))
    lmdb.open_for_write()

    while True:
        data_fc = queue_fc.get()
        is_end = data_fc['is_end']
        if is_end == 1:
            break

        batch_num = data_fc['batch_num']
        ids = data_fc['ids']
        fc_feats = data_fc['fc_feats']

        np_fc_feats = fc_feats.cpu().numpy()
        for j in range(batch_num):
            np_fc_feat = np_fc_feats[j]
            output = io.BytesIO()
            # np.savez(output, x=I) # 921788
            np.savez_compressed(output, x=np_fc_feat)  # 726509
            content = output.getvalue()
            lmdb.insert(str(ids[j]), content)

    lmdb.commit()
    lmdb.close()

    print('lmdb end')


def main(opt):

    preprocess = trn.Compose([trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print('prepro_feature loading json file: ', opt.input_json)
    info = json.load(open(opt.input_json))
    images_root = opt.images_root
    output_lmdb = opt.output_lmdb

    model_cnn = models.setup_default_cnn(opt)
    model_cnn.cuda()
    model_cnn.eval()

    print('setup default cnn')

    batch_num = 0

    len_images = len(info['images'])

    start = None

    thread_cnn = threading.Thread(target=thread_cnn_worker, args=(model_cnn,))
    thread_cnn.setDaemon(True)
    thread_cnn.start()

    thread_lmdb = threading.Thread(target=thread_lmdb_worker, args=(output_lmdb,))
    thread_lmdb.setDaemon(True)
    thread_lmdb.start()

    img_batch = torch.FloatTensor(opt.batch_size, 3, opt.image_size, opt.image_size)
    img_ids = []

    for i in range(len_images):

        img_info = info['images'][i]
        id = img_info['id']
        split = img_info['split']
        file_path = img_info['file_path']

        # print(id, file_path, split)

        full_path = os.path.join(images_root, file_path)

        if opt.verbose == 1:
            print("%d/%d, %s" % (i, len_images, full_path))

        I = imread(full_path)

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        try:
            Ir = imresize(I, (opt.image_size, opt.image_size))
        except:
            print('failed resizing image %s - see http://git.io/vBIE0' % (img_info['file_path']))
            raise

        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.astype('float32') / 255.0
        Ir = Ir.transpose(2, 0, 1)

        img = torch.from_numpy(Ir)
        img = preprocess(img)
        img_batch[i % opt.batch_size] = img
        img_ids.append(id)

        batch_num += 1

        if (i % opt.batch_size == opt.batch_size - 1) or (i == len_images - 1):

            data = {}
            data['img_batch'] = img_batch.cuda().clone()
            data['batch_num'] = batch_num
            data['ids'] = copy.deepcopy(img_ids)
            data['is_end'] = 0
            queue_cnn.put(data)

            batch_num = 0
            img_ids = []
            if start is not None:
                avg_time = (time.time()-start) / opt.batch_size
                print("%d/%d, time:%.3fs, last time:%.3fs cnn queue size:%d, lmdb queue size:%d" %
                      (i, len_images, avg_time, avg_time * (len_images-i), queue_cnn.qsize(), queue_fc.qsize()))

            start = time.time()


    data = {}
    data['is_end'] = 1
    queue_cnn.put(data)

    thread_cnn.join()
    thread_lmdb.join()

    print('all end')

    # data = np.load(io.BytesIO(content))
    # x = data['x']



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    # http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    parser.add_argument('--input_json', type=str, default='/media/amds/data2/dataset/mscoco/data_coco.json', help='')
    parser.add_argument('--images_root', type=str, default='/media/amds/data3/dataset/mscoco', help='')
    parser.add_argument('--output_lmdb', type=str, default='/media/amds/data3/dataset/mscoco_lmdb', help='')
    parser.add_argument('--verbose', type=int, default=0, help='')

    # cnn model
    parser.add_argument('--cnn_model', type=str,
                        default='resnext_101_64x4d',
                        help='')
    parser.add_argument('--input_cnn_resnext_101_64x4d', type=str,
                        default='/media/amds/data2/dataset/resnet/resnext_101_64x4d.pth',
                        help='path to the cnn')

    # setting
    parser.add_argument('--image_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')

    opt = parser.parse_args()
    print 'parsed input parameters:'
    print json.dumps(vars(opt), indent=2)

    main(opt)