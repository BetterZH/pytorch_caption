import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import torch.nn.init as init
import math
import pytorch_fft.fft as cfft

# https://github.com/jnhwkim/cbp/blob/master/CompactBilinearPooling.lua
# https://github.com/locuslab/pytorch_fft
class CompactBilinearPooling(nn.Module):
    def __init__(self, outputSize, homogeneous):
        super(CompactBilinearPooling, self).__init__()

        self.outputSize = outputSize
        self.homogeneous = homogeneous

    def reset(self):
        self.h1 = torch.Tensor()
        self.h2 = torch.Tensor()
        self.s1 = torch.Tensor()
        self.s2 = torch.Tensor()
        self.y = torch.Tensor()
        self.gradInput = {}
        self.tmp = torch.Tensor()

    def sample(self):
        self.h1.uniform_(0, self.outputSize).ceil_()
        self.h2.uniform_(0, self.outputSize).ceil_()
        self.s1.uniform_(0, 2).floor_().mul_(2).add_(-1)
        self.s2.uniform_(0, 2).floor_().mul_(2).add_(-1)

    def psi(self):
        self.y.zero_()
        batchSize = self.input[0].size(0)
        for i in range(2):
            if self.homogeneous:
                self.y[i].index_add_(1, self.h1, self.s1.repeat(batchSize,1)*self.input[i])
            else:
                if i == 0:
                    self.y[i].index_add_(1, self.h1, self.s1.repeat(batchSize,1)*self.input[i])
                else:
                    self.y[i].index_add_(1, self.h2, self.h2.repeat(batchSize,1)*self.input[i])


    def conv(self, x, y):
        batchSize = x.size(0)
        dim = x.size(1)

        x_i = torch.FloatTensor(x.size()).cuda().zero_()
        y_i = torch.FloatTensor(x.size()).cuda().zero_()

        x1_r, x1_i = cfft.fft(x, x_i)
        y1_r, y1_i = cfft.fft(y, y_i)

        return cfft.ifft(x1_r * y1_r, x1_i * y1_i)


    def forward(self, input):

        self.input = input

        inputSizes1 = input[0].size()
        inputSizes2 = input[1].size()

        if len(self.h1.size()) == 0:
            self.h1.resize_(inputSizes1[-1])
            self.h2.resize_(inputSizes2[-1])
            self.s1.resize_(inputSizes1[-1])
            self.s2.resize_(inputSizes2[-1])
            self.sample()

        batchSize = inputSizes1[0]
        self.y.resize_(2, batchSize, self.outputSize)
        self.psi()

        output = self.conv(self.y[0], self.y[1])

        return output



