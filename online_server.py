from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle

import torch
from torch.autograd import Variable

import misc.utils as utils
import models
import zipfile
from data.DataLoaderRaw import *
import itchat
import time
from itchat.content import *
import urllib2,urllib
import cv2

def parse_args():
    # Input arguments and options
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--batch_size', type=int, default=1,
                    help='if > 0 then overrule, otherwise load from checkpoint.')

    # Sampling options
    parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

    # For evaluation on a folder of images:
    parser.add_argument('--start_from_best', type=str, default='/home/amds/caption/caption_checkpoint_best/aic',
                        help="")
    parser.add_argument('--id', type=str, default='aic_weight_2',
                    help='')

    return parser.parse_args()

def get_batch(filepaths, batch_size):

    image_size = 224
    img_batch = np.ndarray([batch_size, 3, image_size, image_size], dtype='float32')

    for i in range(batch_size):

        img = imread(filepaths[i])
        img = imresize(img, (image_size, image_size))

        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
            img = np.concatenate((img,img,img),axis=2)

        img = img.transpose(2, 0, 1)
        img_batch[i] = img[:3]

    data = {}
    data['images'] = img_batch

    return data

def eval_split(model_cnn, model, filepaths, ix_to_word, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 1)

    predictions = []

    data = get_batch(filepaths, batch_size)

    images = torch.from_numpy(data['images']).cuda()
    images = utils.prepro_norm(images, False)
    images = Variable(images, requires_grad=False)

    if models.is_only_fc_feat(caption_model):
        fc_feats = model_cnn(images)
    else:
        fc_feats, att_feats = model_cnn(images)

    if models.is_only_fc_feat(caption_model):
        seq, _ = model.sample(fc_feats, {'beam_size': beam_size})
    else:
        seq, _ = model.sample(fc_feats, att_feats, {'beam_size': beam_size})

    # sents
    sents = utils.decode_sequence(ix_to_word, seq)

    for k, sent in enumerate(sents):
        print(sent)
        sent = ''.join(sent.split())
        predictions.append(sent)

    return predictions

def load_infos(opt):
    infos = {}
    if opt.start_from_best is not None and len(opt.start_from_best) > 0:
        print("start best from %s" % (opt.start_from_best))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from_best, opt.id + '_infos_best.pkl')) as f:
            infos = cPickle.load(f)
    elif opt.start_from is not None and len(opt.start_from) > 0:
        print("start from %s" % (opt.start_from))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, opt.id + '_infos_.pkl')) as f:
            infos = cPickle.load(f)
    return infos

def load_model():
    opt = parse_args()

    # Load infos
    infos = load_infos(opt)

    ignore = ["id", "batch_size", "beam_size", "start_from_best"]
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    print(opt)

    # Setup the model
    model_cnn = models.setup_cnn(opt)
    model_cnn.cuda()

    model = models.setup(opt)
    model.cuda()

    # Make sure in the evaluation mode
    model_cnn.eval()
    model.eval()

    ix_to_word = infos['vocab']

    return model_cnn, model, ix_to_word, opt


def main():

    model_cnn, model, ix_to_word, opt = load_model()

    filepaths = ['/home/amds/Gzcd-fykusez0076319.jpg']
    # Set sample options

    # predictions = eval_split(model_cnn, model, filepaths, ix_to_word,  vars(opt))

    @itchat.msg_register(TEXT)
    def text_reply(msg):
        info = msg['Text'].encode('UTF-8')
        url = 'http://www.tuling123.com/openapi/api'
        data = {'key': '0ca3121447deed5f7848b5c1544d80ed', 'info': info, 'userid': msg.FromUserName}
        data = urllib.urlencode(data)

        url2 = urllib2.Request(url, data)
        response = urllib2.urlopen(url2)

        apicontent = response.read()
        s = json.loads(apicontent, encoding='utf-8')
        print(s)
        if s['code'] == 100000:
            return s['text']

    @itchat.msg_register('Friends')
    def add_friend(msg):
        itchat.add_friend(**msg['Text'])
        itchat.get_contact()
        itchat.send_msg('Nice to meet you!', msg['RecommendInfo']['UserName'])

    @itchat.msg_register(['Picture'])
    def download_files(msg):
        print(msg)
        print(msg.fileName)
        filepath = os.path.join('/home/amds/imgs',msg.fileName)
        # msg.download(filepath)
        msg['Text'](filepath)
        im = cv2.imread(filepath)
        new_path = filepath.split(".")[0] + ".jpg"
        print(new_path)
        cv2.imwrite(new_path, im)
        filepaths = [new_path]
        predictions = eval_split(model_cnn, model, filepaths, ix_to_word, vars(opt))
        print(predictions)
        if len(predictions) > 0:
            return predictions[0]
            # itchat.send(predictions[0], msg.FromUserName)

    @itchat.msg_register('Picture', isGroupChat=True)
    def download_files1(msg):
        print(msg)
        print(msg.User)
        print(msg.fileName)
        filepath = os.path.join('/home/amds/imgs',msg.fileName)
        msg.download(filepath)
        filepaths = [filepath]
        predictions = eval_split(model_cnn, model, filepaths, ix_to_word, vars(opt))
        print(predictions)
        if len(predictions) > 0:
            # return predictions[0]
            itchat.send(predictions[0], msg.FromUserName)

    itchat.auto_login(True, enableCmdQR=2)
    itchat.run()


if __name__ == '__main__':
    main()