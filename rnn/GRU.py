import torch.nn as nn
import torch.nn.functional as F
import torch
import mixer.ReinforceSampler as ReinforceSampler
from torch.autograd import *
import torch.nn.init as init
import math


class GRU_DOUBLE_ATT_STACK_PARALLEL_DROPOUT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(GRU_DOUBLE_ATT_STACK_PARALLEL_DROPOUT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_parallels = num_parallels
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout

        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.a2hs = nn.ModuleList()
        self.i2hs = nn.ModuleList()
        self.h2hs = nn.ModuleList()

        self.a2h1s = nn.ModuleList()
        self.i2h1s = nn.ModuleList()
        self.h2h1s = nn.ModuleList()

        for i in range(self.num_parallels):

            # rnn_size * 2 * batch_size
            a2h = nn.Linear(self.rnn_size, 2 * self.rnn_size)
            i2h = nn.Linear(self.input_size, 2 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 2 * self.rnn_size)

            init.xavier_normal(a2h.weight)
            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)

            self.a2hs.append(a2h)
            self.i2hs.append(i2h)
            self.h2hs.append(h2h)

            # rnn_size * 1 * batch_size
            a2h1 = nn.Linear(self.rnn_size, self.rnn_size)
            i2h1 = nn.Linear(self.input_size, self.rnn_size)
            h2h1 = nn.Linear(self.rnn_size, self.rnn_size)

            init.xavier_normal(a2h1.weight)
            init.xavier_normal(i2h1.weight)
            init.xavier_normal(h2h1.weight)

            self.a2h1s.append(a2h1)
            self.i2h1s.append(i2h1)
            self.h2h1s.append(h2h1)


        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        self.proj = nn.Linear(self.rnn_size, self.output_size)


        # init parameters
        init.xavier_normal(self.a2a.weight)
        init.xavier_normal(self.h2a.weight)
        init.xavier_normal(self.d2d.weight)

        init.xavier_normal(self.a2a1.weight)
        init.xavier_normal(self.h2a1.weight)
        init.xavier_normal(self.d2d1.weight)

        init.xavier_normal(self.proj.weight)

        # batch_size * rnn_size


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        for i in range(self.num_layers):

            prev_h = inputs[i*2]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            # ##################################################
            # spatial attention start
            # (batch * att_size) * rnn_size
            prev_att_v = att.view(-1, self.rnn_size)
            # (batch * att_size) * att_size
            prev_att_v = self.a2a(prev_att_v)
            prev_att_v = F.dropout(prev_att_v, self.dropout)
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            prev_att_h = F.dropout(prev_att_h, self.dropout)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = prev_att_v + prev_att_h_1
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = F.dropout(prev_dot, self.dropout)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            prev_att_res = F.dropout(prev_att_res, self.dropout)
            # spatial attention end
            # ##################################################

            all_next_h = []
            for j in range(self.num_parallels):
                # ##################################################
                # lstm core
                # batch_size * 2*rnn_size

                all_input_sums = self.i2hs[j](xt) + self.h2hs[j](prev_h) + self.a2hs[j](prev_att_res)

                # batch_size * 3*rnn_size
                sigmoid_chunk = all_input_sums[:,:self.rnn_size*2]
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)

                z_gate = sigmoid_chunk[:,0:self.rnn_size]
                r_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]

                # batch_size * 1*rnn_size
                h_gate = F.tanh(self.i2h1s[j](xt) + self.h2h1s[j](r_gate * prev_h) + self.a2h1s[j](prev_att_res))

                next_h = (1 - z_gate) * prev_h + z_gate * h_gate

                next_h = F.dropout(next_h, self.dropout)

                # ##################################################

                all_next_h.append(next_h)

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()

            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            att_v = F.dropout(att_v, self.dropout)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            att_h = F.dropout(att_h, self.dropout)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
            dot = F.dropout(dot, self.dropout)
            dot = dot.view(-1, self.att_size)

            weight = F.softmax(dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1, 2)
            # batch_size * rnn_size
            att_res = torch.bmm(att_seq_t, weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(top_h)

        # policy
        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft