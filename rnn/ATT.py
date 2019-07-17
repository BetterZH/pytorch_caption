import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import torch.nn.init as init
import math

class att_with_spp(nn.Module):
    def __init__(self, rnn_size, spp_size):
        super(att_with_spp, self).__init__()
        self.rnn_size = rnn_size
        self.spp_size = spp_size

        self.pool = nn.AdaptiveAvgPool2d(self.spp_size)
        self.atten = lstm_att_with_att_h(self.rnn_size, self.spp_size*self.spp_size)

    # att_v: batch * rnn_size * pool_size * pool_size
    # h    : batch * rnn_size
    def forward(self, att_v, h):

        # batch * rnn_size * pool_size * pool_size ->
        # batch * rnn_size * spp_size * spp_size
        att_v_1 = self.pool(att_v)

        # batch * rnn_size * (spp_size * spp_size) ->
        # batch * (spp_size * spp_size) * rnn_size
        att_v_2 = att_v_1.view(att_v.size(0), self.rnn_size, -1).transpose(1, 2).contiguous()

        # batch * (spp_size * spp_size) * rnn_size ->
        # batch_size * rnn_size
        res_att = self.atten(att_v_2, h)

        # batch_size * rnn_size
        return res_att


class lstm_att_with_att_h_spp(nn.Module):

    def __init__(self, rnn_size, att_size, pool_size, spp_num):
        super(lstm_att_with_att_h_spp, self).__init__()

        self.rnn_size = rnn_size
        self.att_size = att_size
        self.pool_size = pool_size
        self.spp_num = spp_num

        self.attens = nn.ModuleList()
        for i in range(self.spp_num):
            spp_size = pool_size - i
            # batch * att_size * rnn_size
            atten = att_with_spp(rnn_size, spp_size)
            self.attens.append(atten)


    # att : batch * att_size * rnn_size
    # h   : batch * rnn_size
    def forward(self, att, h):

        # batch * rnn_size * pool_size * pool_size
        att_v = att.transpose(1, 2).contiguous().view(att.size(0), self.rnn_size, self.pool_size, self.pool_size)

        res_atts = []
        for i in range(self.spp_num):
            # batch_size * rnn_size
            res_att = self.attens[i](att_v, h)
            res_atts.append(res_att)

        # batch_size * rnn_size
        res_att = torch.cat([_.unsqueeze(1) for _ in res_atts], 1).mean(1).view(h.size(0), h.size(1))

        # batch_size * rnn_size
        return res_att


class lstm_att_with_att_h(nn.Module):

    def __init__(self, rnn_size, att_size):
        super(lstm_att_with_att_h, self).__init__()

        self.rnn_size = rnn_size
        self.att_size = att_size

        self.a2a = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2a = nn.Linear(self.rnn_size, self.rnn_size)
        self.d2d = nn.Linear(self.rnn_size, 1)

    # x   : batch * rnn_size
    # att : batch * att_size * rnn_size
    # h   : batch * rnn_size
    def forward(self, att, h):

        # ##################################################
        # spatial attention start

        # att_v_1
        # (batch * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * rnn_size
        att_v_1 = self.a2a(att_v)
        # batch * att_size * rnn_size
        att_v_1 = att_v_1.view(-1, self.att_size, self.rnn_size)

        # att_h_1
        # batch * rnn_size
        att_h = self.h2a(h)
        # batch * att_size * rnn_size
        att_h_1 = att_h.unsqueeze(1).expand_as(att_v_1)

        # batch * att_size * rnn_size
        dot = att_v_1 + att_h_1
        # batch * att_size * rnn_size
        dot = F.tanh(dot)
        # (batch * att_size) * rnn_size
        dot = dot.view(-1, self.rnn_size)
        # (batch * att_size) * 1
        dot = self.d2d(dot)
        # batch * att_size
        dot = dot.view(-1, self.att_size)

        # batch_size * att_size * 1
        weight = F.softmax(dot).unsqueeze(2)

        # batch_size * rnn_size * att_size
        att_t = att.transpose(1, 2)

        # batch_size * rnn_size * 1 -> batch_size * rnn_size
        att_res = torch.bmm(att_t, weight).view(-1, self.rnn_size)
        # spatial attention end
        # ##################################################

        # batch_size * rnn_size
        return att_res


class lstm_att_c_with_att_h(nn.Module):

    def __init__(self, rnn_size, att_size):
        super(lstm_att_c_with_att_h, self).__init__()

        self.rnn_size = rnn_size
        self.att_size = att_size

        self.a2a = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2a = nn.Linear(self.rnn_size, self.rnn_size)
        self.d2d = nn.Linear(self.rnn_size, self.rnn_size)

    # att : batch * att_size * rnn_size
    # h   : batch * rnn_size
    def forward(self, att, h):

        # ##################################################
        # channels attention start

        # batch * rnn_size <- batch * att_size * rnn_size
        att_v = att.mean(1)
        # batch * rnn_size
        att_v_1 = self.a2a(att_v)

        # att_h_1
        # batch * rnn_size
        att_h_1 = self.h2a(h)

        # batch * rnn_size
        dot = att_v_1 + att_h_1
        # batch * rnn_size
        dot = F.tanh(dot)
        # batch * rnn_size
        dot = self.d2d(dot)

        # batch_size * rnn_size
        weight = F.softmax(dot)

        # batch_size * att_size * rnn_size
        weight = weight.unsqueeze(1).expand_as(att)

        # batch_size * att_size * rnn_size
        att_res = att * weight

        # channels attention end
        # ##################################################

        # batch_size * att_size * rnn_size
        return att_res


class lstm_att_with_x_att_h(nn.Module):

    def __init__(self, rnn_size, att_size):
        super(lstm_att_with_x_att_h, self).__init__()

        self.rnn_size = rnn_size
        self.att_size = att_size

        self.a2a = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2a = nn.Linear(self.rnn_size, self.rnn_size)
        self.i2a = nn.Linear(self.rnn_size, self.rnn_size)
        # self.a2z = nn.Linear(self.rnn_size, self.rnn_size)
        self.d2d = nn.Linear(self.rnn_size, 1)

    # x   : batch * rnn_size
    # att : batch * att_size * rnn_size
    # h   : batch * rnn_size
    def forward(self, x, att, h):

        # ##################################################
        # spatial attention start

        # att_v_1
        # (batch * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * rnn_size
        att_v_1 = self.a2a(att_v)
        # batch * att_size * rnn_size
        att_v_1 = att_v_1.view(-1, self.att_size, self.rnn_size)

        # # att_z
        # # (batch * att_size) * rnn_size
        # att_z = self.a2z(att_v)
        # # batch * att_size * rnn_size
        # att_z = att_z.view(-1, self.att_size, self.rnn_size)

        # att_h_1
        # batch * rnn_size
        att_h = self.h2a(h)
        # batch * att_size * rnn_size
        att_h_1 = att_h.unsqueeze(1).expand_as(att_v_1)

        # att_x_1
        # batch * rnn_size
        att_x = self.i2a(x)
        # batch * att_size * rnn_size
        att_x_1 = att_x.unsqueeze(1).expand_as(att_v_1)

        # batch * att_size * rnn_size
        dot = att_v_1 + att_h_1 + att_x_1
        # batch * att_size * rnn_size
        dot = F.tanh(dot)
        # (batch * att_size) * rnn_size
        dot = dot.view(-1, self.rnn_size)
        # (batch * att_size) * 1
        dot = self.d2d(dot)
        # batch * att_size
        dot = dot.view(-1, self.att_size)

        # batch_size * att_size * 1
        weight = F.softmax(dot).unsqueeze(2)

        # batch_size * rnn_size * att_size
        att_t = att.transpose(1, 2)

        # batch_size * rnn_size
        att_res = torch.bmm(att_t, weight).view(-1, self.rnn_size)
        # spatial attention end
        # ##################################################

        return att_res
