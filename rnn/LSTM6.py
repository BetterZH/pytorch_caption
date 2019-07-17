import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import torch.nn.init as init
import math
import CORE
import ATT
import dmn.DMN as DMN
import time


# with n num_layers
# new
class LSTM_WITH_TOP_DOWN_ATTEN(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, bu_size, bu_num, dropout):
        super(LSTM_WITH_TOP_DOWN_ATTEN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_parallels = num_parallels
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.bu_size = bu_size
        self.bu_num = bu_num
        self.dropout = dropout

        # core
        self.core1 = CORE.lstm_core(self.input_size, self.rnn_size)

        self.core2 = CORE.lstm_core_with_att(self.input_size, self.rnn_size)

        # bu attention
        self.att = ATT.lstm_att_with_att_h(self.rnn_size, self.bu_size)

        # proj
        self.proj = nn.Linear(self.rnn_size, self.output_size)

        init.xavier_normal(self.proj.weight)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # bu     : batch_size * bu_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, bu, inputs):

        outputs = []

        prev_c_1 = inputs[0]
        prev_h_1 = inputs[1]

        next_c_1, next_h_1 = self.core1(x, prev_c_1, prev_h_1)

        prev_c_2 = inputs[2]
        prev_h_2 = inputs[3]

        prev_bu_res = self.att(bu, next_h_1)

        next_c_2, next_h_2 = self.core2(next_h_1, prev_c_2, prev_h_2, prev_bu_res)

        outputs.append(next_c_1)
        outputs.append(next_h_1)
        outputs.append(next_c_2)
        outputs.append(next_h_2)

        top_h = outputs[-1]

        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)

        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft