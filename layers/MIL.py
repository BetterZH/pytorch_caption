import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MIL(nn.Module):
    def __init__(self, mil_type):
        super(MIL, self).__init__()
        # 0 max
        # 1 nor
        # 2 mean
        self.mil_type = mil_type

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