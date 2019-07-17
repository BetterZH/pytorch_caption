from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle

import torch
from torch.autograd import Variable

import misc.utils as utils
import models
from data.DataLoaderRaw import *
from data.DataLoaderThreadNew import *
from data.DataLoaderThreadBu import *

import random
import sys
from json import encoder
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import *

import misc.Embed as Embed
import models
import rnn.LSTM as LSTM
import rnn.rnn_utils as rnn_utils

from cnn import resnet
from cnn import resnet200
from cnn import resnext_101_32x4d
from cnn import resnext_101_32x4d_7
from cnn import resnext_101_64x4d
from cnn import inceptionresnetv2
from cnn import inceptionv4
from misc.BiShowAttenTellModel import BiShowAttenTellModel
from misc.DoubleAttenMModel import DoubleAttenMModel
from misc.DoubleAttenSCModel import DoubleAttenSCModel
from misc.ITModel import ITModel
from misc.MixerModel import MixerModel
from misc.MoreAttenModel import MoreAttenModel
from misc.SCSTModel import SCSTModel
from misc.ShowAttenTellModel import ShowAttenTellModel
from misc.ShowTellModel import ShowTellModel
from misc.ShowTellPhraseModel import ShowTellPhraseModel
from misc.MoreSupModel import MoreSupModel
from misc.MoreSupPhraseModel import MoreSupPhraseModel
from misc.ShowAttenTellPhraseModel import ShowAttenTellPhraseModel
from misc.MoreSupWeightModel import MoreSupWeightModel
from misc.ShowAttenTellPhraseBuModel import ShowAttenTellPhraseBuModel
from misc.TopDownAttenModel import TopDownAttenModel

def parse_args():

    # Input arguments and options
    parser = argparse.ArgumentParser()

    # the paths of models at different iteration
    parser.add_argument('--start_from_best', type=str, default='/home/amds/caption/caption_checkpoint_best/aic2',
                    help="")

    parser.add_argument('--iterations', type=list, default=[10000, 20000],
                        help="")

    # the path of the ensemble model
    parser.add_argument('--ensemble_model_path', type=str, default='/home/amds/caption/caption_checkpoint_best/aic2',
                        help="")

    # the id of the model
    parser.add_argument('--id', type=str, default='aic_weight_up_1',
                        help="")

    return parser.parse_args()

def save_model_best(model, model_cnn, opt):

    if not os.path.exists(opt.ensemble_model_path):
        os.makedirs(opt.ensemble_model_path)

    checkpoint_path = os.path.join(opt.ensemble_model_path, opt.id + '_model_best_ensemble.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    checkpoint_path_cnn = os.path.join(opt.ensemble_model_path, opt.id + '_model_cnn_best_ensemble.pth')
    torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
    print("model cnn saved to {}".format(checkpoint_path_cnn))


def load_infos(opt, iteration):
    infos = {}
    if opt.start_from_best is not None and len(opt.start_from_best) > 0:
        print("start best from %s" % (opt.start_from_best))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from_best, opt.id + '_infos_best_' + str(iteration) + '.pkl')) as f:
            infos = cPickle.load(f)
    elif opt.start_from is not None and len(opt.start_from) > 0:
        print("start from %s" % (opt.start_from))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, opt.id + '_infos_' + str(iteration) + '.pkl')) as f:
            infos = cPickle.load(f)
    return infos

def setup(opt, iteration):

    if opt.caption_model == 'ShowTell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'ShowAttenTell':
        model = ShowAttenTellModel(opt)
    elif opt.caption_model == 'DoubleAttenSC':
        model = DoubleAttenSCModel(opt)
    elif opt.caption_model == 'Mixer':
        model = MixerModel(opt)
    elif opt.caption_model == 'SCST':
        model = SCSTModel(opt)
    elif opt.caption_model == 'BiShowAttenTellModel':
        model = BiShowAttenTellModel(opt)
    elif opt.caption_model == 'DoubleAttenMModel':
        model = DoubleAttenMModel(opt)
    elif opt.caption_model == 'ITModel':
        model = ITModel(opt)
    elif opt.caption_model == 'ShowTellPhraseModel':
        model = ShowTellPhraseModel(opt)
    elif opt.caption_model == 'MoreAttenModel':
        model = MoreAttenModel(opt)
    elif opt.caption_model == 'MoreSupModel':
        model = MoreSupModel(opt)
    elif opt.caption_model == 'MoreSupPhraseModel':
        model = MoreSupPhraseModel(opt)
    elif opt.caption_model == 'ShowAttenTellPhraseModel' or opt.caption_model == 'ShowAttenTellPhraseRegionModel':
        model = ShowAttenTellPhraseModel(opt)
    elif opt.caption_model == 'MoreSupWeightModel':
        model = MoreSupWeightModel(opt)
    elif opt.caption_model == 'ShowAttenTellPhraseBuModel':
        model = ShowAttenTellPhraseBuModel(opt)
    elif opt.caption_model == 'TopDownAttenModel':
        model = TopDownAttenModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    start_from_best = vars(opt).get('start_from_best', None)
    start_from = vars(opt).get('start_from', None)
    if start_from_best is not None and len(start_from_best) > 0:
        path_model = os.path.join(start_from_best, opt.id + '_model_best_' + str(iteration) + '.pth')
        print(path_model)

        pretrained_dict = torch.load(path_model)
        pretrained_dict = {models.clean_key(k): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()

        # proj_weight = torch.load('/home/amds/att_weight_gram/soft_att_weight_1_model_best.pth')
        # print(proj_weight.keys())
        # pretrained_dict['core.proj_weight.weight'] = proj_weight['core.proj_weight.weight']
        # pretrained_dict['core.proj_weight.bias'] = proj_weight['core.proj_weight.bias']

        # for k, v in pretrained_dict.items():
        #     print(k, v.size(), model_dict[k].size(), v.size(), model_dict[k].size()==v.size())

        for k, v in pretrained_dict.items():
            if not model_dict.has_key(k) or model_dict[k].size() != pretrained_dict[k].size():
                print(k)
                del pretrained_dict[k]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif start_from is not None and len(start_from) > 0:
        path_model = os.path.join(start_from, opt.id + '_model_' + str(iteration) + '.pth')
        print(path_model)
        pretrained_dict = torch.load(path_model)
        pretrained_dict = {models.clean_key(k): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def setup_cnn(opt, iteration):

    if models.is_pre_get(opt.cnn_model):
        print('pre get')
        model_cnn = models.load_cnn_model_identity(opt)
        return model_cnn
    elif opt.cnn_model == "resnet_152":
        model = resnet.resnet152()
        input_cnn = opt.input_cnn_resnet152
    elif opt.cnn_model == "resnet_200":
        model = resnet200.resnet200
        input_cnn = opt.input_cnn_resnet200
    elif opt.cnn_model == "resnext_101_32x4d":
        model = resnext_101_32x4d.resnext_101_32x4d
        input_cnn = opt.input_cnn_resnext_101_32x4d
    elif opt.cnn_model == "resnext_101_32x4d_7":
        model = resnext_101_32x4d_7.resnext_101_32x4d
        input_cnn = opt.input_cnn_resnext_101_32x4d
    elif opt.cnn_model == "resnext_101_64x4d":
        model = resnext_101_64x4d.resnext_101_64x4d
        input_cnn = opt.input_cnn_resnext_101_64x4d
    elif opt.cnn_model == "inceptionresnetv2":
        model = inceptionresnetv2.inceptionresnetv2()
        input_cnn = opt.input_cnn_inceptionresnetv2
    elif opt.cnn_model == "inceptionv4":
        model = inceptionv4.inceptionv4()
        input_cnn = opt.input_cnn_inceptionv4
    else:
        raise Exception("cnn model not supported: {}".format(opt.cnn_model))


    start_from_best = vars(opt).get('start_from_best', None)
    start_from = vars(opt).get('start_from', None)
    if start_from_best is not None and len(start_from_best) > 0:
        model_cnn = models.load_cnn_model(model, opt)
        model_path = os.path.join(start_from_best, opt.id + '_model_cnn_best_' + str(iteration) + '.pth')
        print(model_path)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {models.clean_key(k): v for k, v in pretrained_dict.items()}
        model_cnn.load_state_dict(pretrained_dict)
    elif start_from is not None and len(start_from) > 0:
        model_cnn = models.load_cnn_model(model, opt)
        model_path = os.path.join(opt.start_from, opt.id + '_model_cnn_' + str(iteration) + '.pth')
        print(model_path)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {models.clean_key(k): v for k, v in pretrained_dict.items()}
        model_cnn.load_state_dict(pretrained_dict)
    else:

        print(input_cnn)

        pretrained_dict = torch.load(input_cnn)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        model_cnn = models.load_cnn_model(model, opt)

    return model_cnn


def load_models(opt):

    models = []
    model_cnns = []

    for iteration in opt.iterations:
        print(iteration)

        # Load infos
        infos = load_infos(opt, iteration)

        ignore = ["id", "batch_size", "beam_size", "start_from_best", "input_json",
                  "input_h5", "input_anno", "images_root", "aic_caption_path", "input_bu"]

        for k in vars(infos['opt']).keys():
            if k not in ignore:
                vars(opt).update({k: vars(infos['opt'])[k]})

        # Setup cnn model
        model_cnn = setup_cnn(opt, iteration)
        # Setup model
        model = setup(opt, iteration)

        models.append(model)
        model_cnns.append(model_cnn)

    return models, model_cnns

def ensemble_model(models, model_cnns):

    model_len = len(models)
    model_first = models[0]
    model_dict = model_first.state_dict()

    for k, v in model_dict.items():
        for i in range(model_len-1):
            model_dict[k] += models[i + 1].state_dict()[k]
        model_dict[k] /= model_len

    model_first.load_state_dict(model_dict)

    model_cnn_len = len(model_cnns)
    model_cnn_first = model_cnns[0]
    model_cnn_dict = model_cnn_first.state_dict()

    for k, v in model_cnn_dict.items():
        for i in range(model_cnn_len - 1):
            model_cnn_dict[k] += model_cnns[i + 1].state_dict()[k]
        model_cnn_dict[k] /= model_cnn_len

    model_cnn_first.load_state_dict(model_cnn_dict)

    return model_first, model_cnn_first


def main():
    opt = parse_args()
    # load all models at different iteration
    models, model_cnns = load_models(opt)
    # ensemble all models
    model, model_cnn = ensemble_model(models, model_cnns)
    # save the best model
    save_model_best(model, model_cnn, opt)

if __name__ == '__main__':
    main()


