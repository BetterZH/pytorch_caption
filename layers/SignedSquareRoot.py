import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import torch.nn.init as init
import math

# https://github.com/jnhwkim/cbp/blob/master/SignedSquareRoot.lua
class SignedSquareRoot(nn.Module):
    def __init__(self):
        super(SignedSquareRoot, self).__init__()


    def forward(self, input):

        output = input.abs().sqrt() * input.sign()

        return output


