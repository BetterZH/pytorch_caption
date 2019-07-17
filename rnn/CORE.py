import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import torch.nn.init as init
import math

class lstm_core_with_att(nn.Module):
    def __init__(self, input_size, rnn_size):
        super(lstm_core_with_att, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

    def forward(self, xt, prev_c, prev_h, att_res):
        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(xt) + self.h2h(prev_h) + self.a2h(att_res)

        # batch_size * 4*rnn_size
        sigmoid_chunk = all_input_sums[:, :3 * self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:, 0:self.rnn_size]
        forget_gate = sigmoid_chunk[:, self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:, self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:, self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)

        # batch_size * rnn_size
        next_h = out_gate * F.tanh(next_c)
        # ##################################################

        return next_c, next_h

class lstm_core(nn.Module):
    def __init__(self, input_size, rnn_size):
        super(lstm_core, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

    def forward(self, xt, prev_c, prev_h):
        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(xt) + self.h2h(prev_h)

        # batch_size * 4*rnn_size
        sigmoid_chunk = all_input_sums[:, :3 * self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:, 0:self.rnn_size]
        forget_gate = sigmoid_chunk[:, self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:, self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:, self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)

        # batch_size * rnn_size
        next_h = out_gate * F.tanh(next_c)
        # ##################################################

        return next_c, next_h


class lstm_linear(nn.Module):

    def __init__(self, rnn_size, block_num):
        super(lstm_linear, self).__init__()

        self.layer = self._make_layer(rnn_size, block_num)

    def _make_layer(self, rnn_size, block_num):

        layers = []

        for i in range(block_num-1):
            linear = nn.Linear(rnn_size, rnn_size)

            # init weight
            init.xavier_normal(linear.weight)

            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())


        layers.append(nn.Linear(rnn_size, rnn_size))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.layer(x)

        return x


class lstm_mul_linear_core_with_att(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, block_num):
        super(lstm_mul_linear_core_with_att, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.block_num = block_num

        # core
        self.linears = nn.ModuleList()
        for i in range(12):
            block = lstm_linear(self.rnn_size, self.block_num)
            self.linears.append(block)


    def forward(self, xt, prev_c, prev_h, att_res):
        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        input_gate = F.sigmoid(self.linears[0](xt) + self.linears[1](prev_h) + self.linears[2](att_res))
        forget_gate = F.sigmoid(self.linears[3](xt) + self.linears[4](prev_h) + self.linears[5](att_res))
        output_gate = F.sigmoid(self.linears[6](xt) + self.linears[7](prev_h) + self.linears[8](att_res))
        in_transform = F.tanh(self.linears[9](xt) + self.linears[10](prev_h) + self.linears[11](att_res))

        next_c = forget_gate * prev_c + input_gate * in_transform
        next_h = output_gate * F.tanh(next_c)
        # ##################################################

        return next_c, next_h


class lstm_core_with_att_bu(nn.Module):
    def __init__(self, input_size, rnn_size):
        super(lstm_core_with_att_bu, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.b2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

    def forward(self, xt, prev_c, prev_h, att_res, bu_res):
        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(xt) + self.h2h(prev_h) + self.a2h(att_res) + self.b2h(bu_res)

        # batch_size * 4*rnn_size
        sigmoid_chunk = all_input_sums[:, :3 * self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:, 0:self.rnn_size]
        forget_gate = sigmoid_chunk[:, self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:, self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:, self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)

        # batch_size * rnn_size
        next_h = out_gate * F.tanh(next_c)
        # ##################################################

        return next_c, next_h