import torch
import sys
# sys.path.append("/media/amds/data1/code_caption/tensorflow-model-zoo.torch")
# from inceptionv4.pytorch_load import inceptionv4
# net = inceptionv4()
# input = torch.autograd.Variable(torch.ones(1,3,299,299))
# output = net.forward(input)
# print(output.size())

# from inceptionresnetv2.pytorch_load import inceptionresnetv2
# net = inceptionresnetv2()
# input = torch.autograd.Variable(torch.ones(1,3,299,299))
# output = net.forward(input)
# print(output.size())

import inceptionresnetv2
import inceptionv4

def clean_key(key):
    if key.startswith('module.'):
        key = key.partition('module.')[2]
    return key

# (12L, 1536L, 8L, 8L)

# 1237M
model = inceptionresnetv2.inceptionresnetv2()
model_path = "/media/amds/data2/dataset/inception/inceptionresnetv2-d579a627.pth"

# 955M
# model = inceptionv4.inceptionv4()
# model_path = "/media/amds/data2/dataset/inception/inceptionv4-97ef9c30.pth"

pretrained_dict = torch.load(model_path)

model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)

model.cuda()

while True:
    input = torch.autograd.Variable(torch.ones(12,3,299,299)).cuda()
    output = model.forward(input)
    print(output.size())

