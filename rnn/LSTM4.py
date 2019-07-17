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
class LSTM_SOFT_ATT_STACK_PARALLEL_NO_MEMORY(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT_STACK_PARALLEL_NO_MEMORY, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_parallels = num_parallels
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout

        # core
        self.cores = nn.ModuleList()
        for i in range(self.num_layers * self.num_parallels):
            core = CORE.lstm_core_with_att(self.input_size, self.rnn_size)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers):
            att = ATT.lstm_att_with_att_h(self.rnn_size, self.att_size)
            self.attens.append(att)

        # proj
        self.proj = nn.Linear(self.rnn_size, self.output_size)

        # memory
        # self.memory = DMN.DMNCPlus(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        for i in range(self.num_layers):
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):

                next_c, next_h = self.cores[i*self.num_parallels+j](xt, prev_c, prev_h, prev_att_res)

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1)
            next_c = next_c.mean(1).view_as(prev_c)
            # next_c = next_c.max(1)[0].view_as(prev_c)

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1)
            next_h = next_h.mean(1).view_as(prev_h)
            # next_h = next_h.max(1)[0].view_as(prev_h)

            # batch_size * rnn_size
            top_h = next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

        top_h = outputs[-1]

        # print(att.size())
        # print(top_h.size())

        # start = time.time()
        # ouput = self.memory(att, top_h.unsqueeze(1))
        ouput = self.proj(top_h)
        # print("time:",time.time()-start)

        logsoft = F.log_softmax(ouput)

        return outputs, logsoft


# with n num_layers
# new
class LSTM_SOFT_ATT_STACK_PARALLEL_MEMORY(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, num_hop, dropout):
        super(LSTM_SOFT_ATT_STACK_PARALLEL_MEMORY, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_parallels = num_parallels
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout
        self.num_hop = num_hop

        # core
        self.cores = nn.ModuleList()
        for i in range(self.num_layers * self.num_parallels):
            core = CORE.lstm_core_with_att(self.input_size, self.rnn_size)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers):
            att = ATT.lstm_att_with_att_h(self.rnn_size, self.att_size)
            self.attens.append(att)

        # proj
        # self.proj = nn.Linear(self.rnn_size, self.output_size)

        # memory
        self.memory = DMN.DMNCPlus(self.rnn_size, self.output_size, self.num_hop)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        for i in range(self.num_layers):
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):

                next_c, next_h = self.cores[i*self.num_parallels+j](xt, prev_c, prev_h, prev_att_res)

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1)
            next_c = next_c.mean(1).view_as(prev_c)
            # next_c = next_c.max(1)[0].view_as(prev_c)

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1)
            next_h = next_h.mean(1).view_as(prev_h)
            # next_h = next_h.max(1)[0].view_as(prev_h)

            # batch_size * rnn_size
            top_h = next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

        top_h = outputs[-1]

        # print(att.size())
        # print(top_h.size())

        # start = time.time()
        ouput = self.memory(att, top_h.unsqueeze(1))
        # print("time:",time.time()-start)

        logsoft = F.log_softmax(ouput)

        return outputs, logsoft

