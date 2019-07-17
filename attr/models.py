import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import models

import cnn.resnet as resnet
import cnn.resnet200 as resnet200
import cnn.resnext_101_32x4d as resnext_101_32x4d
import cnn.resnext_101_64x4d as resnext_101_64x4d

from torch.autograd import Variable

class MIL(nn.Module):
    def __init__(self):
        super(MIL, self).__init__()
        # 0 max
        # 1 nor
        # 2 mean
        self.mil_type = 2

    # input: batch_size * channels * h * w
    def forward(self, input):

        batch_size = input.size(0)
        channels = input.size(1)

        if self.mil_type == 0:
            # output: batch_size * channels
            output = input.max(3)[0].max(2)[0].view(batch_size, channels)
        elif self.mil_type == 1:
            # prob: batch_size * channels
            prob = 1 - (1 - input).prod(3).prod(2).view(batch_size, channels)
            # max_prob: batch_size * channels
            max_prob = input.max(3)[0].max(2)[0].view(batch_size, channels)
            # output: batch_size * channels
            output = torch.max(prob, max_prob)
        elif self.mil_type == 2:
            # output: batch_size * channels
            output = input.mean(3).mean(2).view(batch_size, channels)

        # output: batch_size * channels
        return output


class resnet_mil(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_mil, self).__init__()

        self.resnet = resnet
        self.conv1 = nn.Conv2d(2048, 1000, 1)
        self.mil = MIL()


    def forward(self, x):

        #   2048 * 7 * 7
        # x batch_size * channels * h * w
        x = self.resnet(x)

        #   1000 * 7 * 7
        # x batch_size * channels * h * w
        x = self.conv1(x)

        # sigmoid
        x = F.sigmoid(x)

        # mil
        x = self.mil(x)

        # batch_size * 1000
        return x

def load_cnn_model(model, opt):

    model_cnn = resnet_mil(model, opt)

    return model_cnn


def setup_cnn(opt):

    if opt.cnn_model == "resnet_152":
        model = resnet.resnet152()
        input_cnn = opt.input_cnn_resnet152
    elif opt.cnn_model == "resnet_200":
        model = resnet200.resnet200
        input_cnn = opt.input_cnn_resnet200
    elif opt.cnn_model == "resnext_101_32x4d":
        model = resnext_101_32x4d.resnext_101_32x4d
        input_cnn = opt.input_cnn_resnext_101_32x4d
    elif opt.cnn_model == "resnext_101_64x4d":
        model = resnext_101_64x4d.resnext_101_64x4d
        input_cnn = opt.input_cnn_resnext_101_64x4d
    else:
        raise Exception("cnn model not supported: {}".format(opt.cnn_model))

    def clean_key(key):
        if key.startswith('module.'):
            key = key.partition('module.')[2]
        return key

    start_from_best = vars(opt).get('start_from_best', None)
    start_from = vars(opt).get('start_from', None)
    if start_from_best is not None and len(start_from_best) > 0:
        model_cnn = load_cnn_model(model, opt)
        model_path = os.path.join(start_from_best, 'model_cnn_' + opt.id + '_best.pth')
        print(model_path)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
        model_cnn.load_state_dict(pretrained_dict)
    elif start_from is not None and len(start_from) > 0:
        model_cnn = load_cnn_model(model, opt)
        model_path = os.path.join(opt.start_from, 'model_cnn_' + opt.id + '.pth')
        print(model_path)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
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

        model_cnn = load_cnn_model(model, opt)

    return model_cnn