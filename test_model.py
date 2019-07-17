import torch
from torch.utils.serialization import load_lua

from cnn import resnet200
from cnn import resnext_101_64x4d


def resnet200_1():
    filename = "/media/amds/data1/model/TorchResnet/resnet-200_cpu.t7"
    model = load_lua(filename)
    print(model)

def load_state(model, filepath):
    pretrained_dict = torch.load(filepath)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def setup_resnet200():
    filepath = "/media/amds/data2/dataset/resnet/resnet_200_cpu.pth"
    model = resnet200.resnet200
    load_state(model, filepath)

    print(model)
    model.cuda()
    # for layer in range(13,10,-1):
    #     model.remove(layer)
    # print(model)
    # model.cuda()
    # print(model)

def setup_resnext_101_64x4d():
    filepath = "/media/amds/data2/dataset/resnet/resnext_101_64x4d.pth"
    model = resnext_101_64x4d.resnext_101_64x4d
    load_state(model, filepath)
    print(model)
    model.cuda()

# setup_resnext_101_64x4d()
setup_resnext_101_64x4d()
