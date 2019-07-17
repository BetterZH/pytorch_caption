import torch
from torch.autograd import Variable
import pytorch_fft.fft.autograd as fft
import numpy as np

f = fft.Fft2d()
invf= fft.Ifft2d()

fx, fy = (Variable(torch.arange(0,100).view((1,1,10,10)).cuda(), requires_grad=True),
          Variable(torch.zeros(1, 1, 10, 10).cuda(),requires_grad=True))
k1,k2 = f(fx,fy)
z = k1.sum() + k2.sum()
z.backward()
print(fx.grad, fy.grad)