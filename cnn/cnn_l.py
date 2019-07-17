import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# input channels 16 * 512
class CNN_L(nn.Module):
    def __init__(self):
        super(CNN_L, self).__init__()

        block = BasicBlock
        layers = [2, 2]

        self.inplanes = 64
        self.conv1 = nn.Conv1d(16, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AvgPool2d(4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class D_L(nn.Module):
    def __init__(self):
        super(D_L, self).__init__()

        self.cnn1 = CNN_L()
        self.cnn2 = CNN_L()
        self.v = nn.Linear(1024, 2)

    # x : batch_size * size
    # y : batch_size * size
    def forward(self, x1, x2):

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)

        # batch_size * 2 * size
        x = torch.cat([_.unsqueeze(1) for _ in [x1, x2]], 1)
        x = x.view(x.size(0), -1)
        x = self.v(x)
        x = F.log_softmax(x)

        return x



if __name__ == '__main__':

    d = D_L()

    x1 = torch.randn(8, 16, 512)
    x1 = Variable(x1, requires_grad=True)

    x2 = torch.randn(8, 16, 512)
    x2 = Variable(x2, requires_grad=True)

    x = d(x1, x2)
    sampleLogprobs, it = x.max(1)
    it = it.view(-1).long()

    print(x)
    print(x.size())
    print(sampleLogprobs, it)