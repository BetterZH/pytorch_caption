from CompactBilinearPooling import  CompactBilinearPooling as CBP
import torch
from torch.autograd import Variable as Variable
import numpy as np
import torch.nn as nn
from ipdb import set_trace as st

m = nn.Linear(1000, 512)
input = Variable(torch.randn(128, 1000))
output = m(input)
print(output.size())

cbp = CBP(512, 512, 1000)
l = nn.Linear(1000, 512)

c = nn.Sequential(cbp.cuda(), l.cuda())

x = np.random.rand(128, 512).astype(np.float32)
y = np.random.rand(128, 512).astype(np.float32)

a = Variable(torch.Tensor(x).cuda(), requires_grad = True)
b = Variable(torch.Tensor(y).cuda(), requires_grad = True)

cbp1 = cbp(a, b)
print('cbp1 size', cbp1.size())

result = l(cbp1)
print('result size', result.size())

# loss = result.sum()
# loss.backward()

# print(cbp.grad)
# print(a.grad)
# print(b.grad)
print("run success")
