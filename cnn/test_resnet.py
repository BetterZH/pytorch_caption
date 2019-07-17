import torch
import torchvision.models as tmodels
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from scipy import misc
from scipy.misc import imread, imresize
import os

if __name__ == '__main__':
    resnet152 = tmodels.resnet152(pretrained=False)
    # resnet152.eval()

    # print(type(resnet152.state_dict()))

    # print(resnet152.state_dict().keys())

    for key,val in resnet152.state_dict().items():
        print(key, type(val), val.size())

    paths = ["COCO_train2014_000000581921.jpg",
             "COCO_train2014_000000291797.jpg",
             "COCO_train2014_000000291788.jpg"]

    # for i in range(len(paths)):
    #     # im_norm = torch.from_numpy(np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5)
    #
    #     im_norm = torch.FloatTensor(1, 3, 224, 224)
    #     I = imread(os.path.join("/media/amds/data3/dataset/mscoco/train2014/", paths[i]))
    #     I = imresize(I, (224, 224))
    #     I = I.astype(np.float32) / 255.0 - 0.5
    #     I = I.transpose(2, 0, 1)
    #     im_norm[0] = torch.from_numpy(I)
    #
    #     data = Variable(im_norm)
    #     prob = resnet152(data).data.numpy()
    #     # print prob[0, ...].argmax()
    #     print prob.argmax()
