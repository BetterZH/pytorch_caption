import torch.nn as nn
import torch.nn.functional as F
import torch
import mixer.ReinforceSampler as ReinforceSampler
from torch.autograd import *
import torch.nn.init as init
import math

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, dropout):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.rnn_size = rnn_size

        self.i2h = nn.Linear(self.input_size, 4*self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4*self.rnn_size)
        self.proj = nn.Linear(self.rnn_size, self.output_size)

        self.init_weight()

    # x      : batch_size * input_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, inputs):

        prev_c = inputs[0]
        prev_h = inputs[1]

        # batch_size * rnn_size -> batch_size * 4*rnn_size
        all_input_sums = self.i2h(x) + self.h2h(prev_h)

        sigmoid_chunk = all_input_sums[:,0:3*self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:,0:self.rnn_size]
        forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)
        next_h = out_gate * F.tanh(next_c)

        outputs = []
        outputs.append(next_c)
        outputs.append(next_h)

        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)

        logsoft = F.log_softmax(self.proj(top_h))
        # logsoft = F.softmax(self.proj(top_h))

        # next_h   : batch_size * rnn_size
        # next_c   : batch_size * rnn_size
        # logsofts : batch_size * (vocab_size + 1)
        return outputs, logsoft

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.rnn_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class LSTM_ATTENTION(nn.Module):
    def __init__(self, rnn_size):
        super(LSTM_ATTENTION, self).__init__()
        self.rnn_size = rnn_size

        # rnn_size -> rnn_size
        self.h2h = nn.Linear(rnn_size, rnn_size)

        # rnn_size -> rnn_size
        self.d2h = nn.Linear(rnn_size, rnn_size)

        # rnn_size -> 1
        self.V = nn.Linear(rnn_size, 1)

        self.C = nn.Linear(rnn_size, rnn_size)

    # h : batch_size * att_size * rnn_size
    # d   : batch_size * rnn_size
    def forward(self, h, d):

        batch_size = h.size(0)
        att_size = h.size(1)
        rnn_size = h.size(2)

        # (batch_size * att_size) * rnn_size
        d = d.unsqueeze(1).expand_as(h).view(-1, self.rnn_size)

        # (batch_size * att_size) * rnn_size
        h_v = h.view(-1, self.rnn_size)

        # (batch_size * att_size)
        u = self.V(F.tanh(self.h2h(h_v) + self.d2h(d)))
        u = u.view(batch_size, att_size)

        # batch_size * rnn_size * att_size
        h_t = h.transpose(1, 2)

        # batch_size * att_size
        a = F.softmax(u)

        # batch_size * rnn_size
        att_res = torch.bmm(h_t, a.unsqueeze(2)).squeeze()

        p = F.softmax(self.C(d + att_res))

        return p


class LSTM_SIMPLE(nn.Module):
    def __init__(self, input_size, rnn_size):
        super(LSTM_SIMPLE, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size

        self.i2h = nn.Linear(self.input_size, 4*self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4*self.rnn_size)

    # x      : batch_size * input_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, inputs):

        prev_c = inputs[0]
        prev_h = inputs[1]

        # batch_size * rnn_size -> batch_size * 4*rnn_size
        all_input_sums = self.i2h(x) + self.h2h(prev_h)

        n1 = all_input_sums[:, 0:self.rnn_size]
        n2 = all_input_sums[:, self.rnn_size:self.rnn_size*2]
        n3 = all_input_sums[:, self.rnn_size*2:self.rnn_size*3]
        n4 = all_input_sums[:, self.rnn_size*3:self.rnn_size*4]

        in_gate = F.sigmoid(n1)
        forget_gate = F.sigmoid(n2)
        out_gate = F.sigmoid(n3)
        in_transform = F.tanh(n4)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)
        next_h = out_gate * F.tanh(next_c)

        outputs = []
        outputs.append(next_c)
        outputs.append(next_h)

        return outputs

class LSTM_ATTEN_LAYER(nn.Module):
    def __init__(self, rnn_size):
        super(LSTM_ATTEN_LAYER, self).__init__()
        self.rnn_size = rnn_size
        self.core = LSTM_SIMPLE(rnn_size, rnn_size)

    # fc_feats : batch_size * rnn_size
    # att_feats : batch_size * att_size * rnn_size
    def forward(self, fc_feats, att_feats):

        batch_size = fc_feats.size(0)

        prev_h, prev_c = (Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda(),
         Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda())

        att_size = att_feats.size(1)
        hiddens = []
        for t in xrange(att_size):
            # batch_size * rnn_size
            xt = att_feats[:,t,:].squeeze()
            prev_h, prev_c = self.core(xt, prev_h, prev_c)
            hiddens.append(prev_h)

        # batch_size * att_size * rnn_size
        h = torch.cat([_.unsqueeze(1) for _ in hiddens],1)

        return h


class LSTM_SOFT_ATT(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_size = att_size

        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)

        self.proj = nn.Linear(self.rnn_size, self.output_size)

    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        prev_c = inputs[0]
        prev_h = inputs[1]

        # (batch * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * att_size
        att_v = self.a2a(att_v)
        # batch * att_size * att_size
        att_v = att_v.view(-1, self.att_size, self.att_size)
        # batch * att_size
        att_h = self.h2a(prev_h)
        # batch * att_size * att_size
        att_h = att_h.unsqueeze(2).expand_as(att_v)

        # batch * att_size * att_size
        dot = att_h + att_v
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_size)
        dot = self.d2d(dot)
        dot = dot.view(-1, self.att_size)

        # batch_size * att_size
        weigth = F.softmax(dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1, 2)
        # batch_size * rnn_size
        att_res = torch.bmm(att_seq_t, weigth.unsqueeze(2)).squeeze()

        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(x) + self.h2h(prev_h) + self.r2a(att_res)

        sigmoid_chunk = all_input_sums[:,0:3*self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:,0:self.rnn_size]
        forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)
        next_h = out_gate * F.tanh(next_c)

        if self.dropout > 0:
            next_h = F.dropout(next_h, self.dropout)

        outputs = []
        outputs.append(next_c)
        outputs.append(next_h)

        logsolft = F.log_softmax(self.proj(next_h))

        return outputs, logsolft



class LSTM_SOFT_ATT_NOX(nn.Module):
    def __init__(self, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT_NOX, self).__init__()

        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout

        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, att, inputs):

        prev_c = inputs[0]
        prev_h = inputs[1]

        # (batch * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * att_size
        att_v = self.a2a(att_v)
        # batch * att_size * att_size
        att_v = att_v.view(-1, self.att_size, self.att_size)
        # batch * att_size
        att_h = self.h2a(prev_h)
        # batch * att_size * att_size
        att_h = att_h.unsqueeze(2).expand_as(att_v)

        # batch * att_size * att_size
        dot = att_h + att_v
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_size)
        dot = self.d2d(dot)
        dot = dot.view(-1, self.att_size)

        # batch_size * att_size
        weigth = F.softmax(dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1, 2)
        # batch_size * rnn_size
        att_res = torch.bmm(att_seq_t, weigth.unsqueeze(2)).squeeze()

        # batch_size * 4*rnn_size
        all_input_sums = self.h2h(prev_h) + self.r2a(att_res)

        sigmoid_chunk = all_input_sums[:,0:3*self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:,0:self.rnn_size]
        forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)
        next_h = out_gate * F.tanh(next_c)

        if self.dropout > 0:
            next_h = F.dropout(next_h, self.dropout)

        outputs = []
        outputs.append(next_c)
        outputs.append(next_h)

        return outputs


class LSTM_SOFT_ATT_STACK(nn.Module):
    def __init__(self, input_size, output_size, num_layers, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT_STACK, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_size = att_size
        self.num_layers = num_layers

        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)

        self.proj = nn.Linear(self.rnn_size, self.output_size)

    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        for i in range(self.num_layers):
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]
            if i > 0:
                x = outputs[-1]

            # (batch * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            # (batch * att_size) * att_size
            att_v = self.a2a(att_v)
            # batch * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            att_h = self.h2a(prev_h)
            # batch * att_size * att_size
            att_h = att_h.unsqueeze(2).expand_as(att_v)

            # batch * att_size * att_size
            dot = att_h + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d(dot)
            dot = dot.view(-1, self.att_size)

            # batch_size * att_size
            weigth = F.softmax(dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1, 2)
            # batch_size * rnn_size
            att_res = torch.bmm(att_seq_t, weigth.unsqueeze(2)).squeeze()

            # batch_size * 4*rnn_size
            all_input_sums = self.i2h(x) + self.h2h(prev_h) + self.r2a(att_res)

            sigmoid_chunk = all_input_sums[:,0:3*self.rnn_size]
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)

            in_gate = sigmoid_chunk[:,0:self.rnn_size]
            forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
            out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

            in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
            in_transform = F.tanh(in_transform)

            next_c = (forget_gate * prev_c) + (in_gate * in_transform)
            next_h = out_gate * F.tanh(next_c)

            outputs.append(next_c)
            outputs.append(next_h)

        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)

        logsolft = F.log_softmax(self.proj(top_h))

        return outputs, logsolft


class LSTM_SOFT_ATT_TOP(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT_TOP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_size = att_size

        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)

    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, prev_h, prev_c):

        # (batch * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * att_size
        att_v = self.a2a(att_v)
        # batch * att_size * att_size
        att_v = att_v.view(-1, self.att_size, self.att_size)
        # batch * att_size
        att_h = self.h2a(prev_h)
        # batch * att_size * att_size
        att_h = att_h.unsqueeze(2).expand_as(att_v)

        # batch * att_size * att_size
        dot = att_h + att_v
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_size)
        dot = self.d2d(dot)
        dot = dot.view(-1, self.att_size)

        # batch_size * att_size
        weigth = F.softmax(dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1, 2)
        # batch_size * rnn_size
        att_res = torch.bmm(att_seq_t, weigth.unsqueeze(2)).squeeze()

        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(x) + self.h2h(prev_h) + self.r2a(att_res)

        sigmoid_chunk = all_input_sums[:,0:3*self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:,0:self.rnn_size]
        forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)
        next_h = out_gate * F.tanh(next_c)

        top_h = next_h
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)

        outputs = []
        outputs.append(next_c)
        outputs.append(next_h)

        return outputs, top_h


class LSTM_DOUBLE_ATT(nn.Module):

    def __init__(self, input_size, output_size, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout

        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        self.proj = nn.Linear(self.rnn_size, self.output_size)


        # init parameters
        # init.xavier_normal(self.a2a.weight)
        # init.xavier_normal(self.h2a.weight)
        # init.xavier_normal(self.d2d.weight)
        #
        # init.xavier_normal(self.a2h.weight)
        # init.xavier_normal(self.i2h.weight)
        # init.xavier_normal(self.h2h.weight)
        #
        # init.xavier_normal(self.a2a1.weight)
        # init.xavier_normal(self.h2a1.weight)
        # init.xavier_normal(self.d2d1.weight)
        #
        # init.xavier_normal(self.proj.weight)

        # batch_size * rnn_size


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, state):

        prev_c = state[0]
        prev_h = state[1]

        # ##################################################
        # spatial attention start
        # (batch * att_size) * rnn_size
        prev_att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * att_size
        prev_att_v = self.a2a(prev_att_v)
        # batch * att_size * att_size
        prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
        # batch * att_size
        prev_att_h = self.h2a(prev_h)
        # batch * att_size * att_size
        prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

        # batch * att_size * att_size
        prev_dot = prev_att_v + prev_att_h_1
        prev_dot = F.tanh(prev_dot)
        prev_dot = prev_dot.view(-1, self.att_size)
        prev_dot = self.d2d(prev_dot)
        prev_dot = prev_dot.view(-1, self.att_size)

        # batch_size * att_size
        prev_weight = F.softmax(prev_dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1,2)
        # batch_size * rnn_size
        prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
        # spatial attention end
        # ##################################################


        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(x) + self.h2h(prev_h) + self.a2h(prev_att_res)

        # batch_size * 4*rnn_size
        sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:,0:self.rnn_size]
        forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)

        # batch_size * rnn_size
        next_h = out_gate * F.tanh(next_c)
        # ##################################################


        # ##################################################
        # spatial attention start
        # (batch_size * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        att_v = self.a2a1(att_v)
        # batch_size * att_size * att_size
        att_v = att_v.view(-1, self.att_size, self.att_size)

        # batch_size * att_size
        att_h = self.h2a1(next_h)
        # batch_size * att_size * att_size
        att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

        # batch_size * att_size * att_size
        dot = att_h_1 + att_v
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_size)
        dot = self.d2d1(dot)
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

        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        outputs = []
        outputs.append(next_c)
        outputs.append(top_h)

        return outputs, logsoft


class LSTM_DOUBLE_ATT_STACK(nn.Module):

    def __init__(self, input_size, output_size, num_layers, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout

        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        self.proj = nn.Linear(self.rnn_size, self.output_size)


        # init parameters
        init.xavier_normal(self.a2a.weight)
        init.xavier_normal(self.h2a.weight)
        init.xavier_normal(self.d2d.weight)

        init.xavier_normal(self.a2h.weight)
        init.xavier_normal(self.i2h.weight)
        init.xavier_normal(self.h2h.weight)

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
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

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
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = torch.add(prev_att_v, prev_att_h_1)
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################


            # ##################################################
            # lstm core
            # batch_size * 4*rnn_size
            all_input_sums = self.i2h(xt) + self.h2h(prev_h) + self.a2h(prev_att_res)

            # batch_size * 4*rnn_size
            sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)

            in_gate = sigmoid_chunk[:,0:self.rnn_size]
            forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
            out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

            in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
            in_transform = F.tanh(in_transform)

            next_c = (forget_gate * prev_c) + (in_gate * in_transform)

            # batch_size * rnn_size
            next_h = out_gate * F.tanh(next_c)
            # ##################################################


            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
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

            outputs.append(next_c)
            outputs.append(top_h)

        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft

class LSTM_DOUBLE_ATT_STACK_PARALLEL(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL, self).__init__()

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

        for i in range(self.num_parallels):

            a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(a2h.weight)
            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)

            self.a2hs.append(a2h)
            self.i2hs.append(i2h)
            self.h2hs.append(h2h)


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
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

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
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = torch.add(prev_att_v, prev_att_h_1)
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                # ##################################################
                # lstm core
                # batch_size * 4*rnn_size
                all_input_sums = self.i2hs[j](xt) + self.h2hs[j](prev_h) + self.a2hs[j](prev_att_res)

                # batch_size * 4*rnn_size
                sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)

                in_gate = sigmoid_chunk[:,0:self.rnn_size]
                forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
                out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

                in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
                in_transform = F.tanh(in_transform)

                next_c = (forget_gate * prev_c) + (in_gate * in_transform)

                # batch_size * rnn_size
                next_h = out_gate * F.tanh(next_c)
                # ##################################################

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()

            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
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

            outputs.append(next_c)
            outputs.append(top_h)

        # policy
        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft


#
class LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT, self).__init__()

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

        for i in range(self.num_layers * self.num_parallels):

            i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(i2h.weight)
            init.xavier_normal(a2h.weight)
            init.xavier_normal(h2h.weight)

            self.i2hs.append(i2h)
            self.a2hs.append(a2h)
            self.h2hs.append(h2h)


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
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

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
            prev_att_v = F.dropout(prev_att_v)
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            prev_att_h = F.dropout(prev_att_h)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = torch.add(prev_att_v, prev_att_h_1)
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                ind = i * self.num_parallels + j
                # ##################################################
                # lstm core
                # batch_size * 4*rnn_size
                all_input_sums = self.i2hs[ind](xt) + self.h2hs[ind](prev_h) + self.a2hs[ind](prev_att_res)

                # batch_size * 4*rnn_size
                sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)

                in_gate = sigmoid_chunk[:,0:self.rnn_size]
                forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
                out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

                in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
                in_transform = F.tanh(in_transform)

                next_c = (forget_gate * prev_c) + (in_gate * in_transform)

                # batch_size * rnn_size
                next_h = out_gate * F.tanh(next_c)
                # ##################################################

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()


            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            att_v = F.dropout(att_v)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            att_h = F.dropout(att_h)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
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

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)

            outputs.append(next_c)
            outputs.append(top_h)

        # policy
        top_h = outputs[-1]
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft

class LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_ITEM(nn.Module):

    def __init__(self, input_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_ITEM, self).__init__()

        self.input_size = input_size
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

        for i in range(self.num_parallels):

            a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            i2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(a2h.weight)
            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)

            self.a2hs.append(a2h)
            self.i2hs.append(i2h)
            self.h2hs.append(h2h)


        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)


        # init parameters
        init.xavier_normal(self.a2a.weight)
        init.xavier_normal(self.h2a.weight)
        init.xavier_normal(self.d2d.weight)

        init.xavier_normal(self.a2a1.weight)
        init.xavier_normal(self.h2a1.weight)
        init.xavier_normal(self.d2d1.weight)



    # x      : batch_size * input_size
    # att    : batch_size * att_size * input_size
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

            # ##################################################
            # spatial attention start
            # (batch * att_size) * input_size
            prev_att_v = att.view(-1, self.rnn_size)
            # (batch * att_size) * att_size
            prev_att_v = self.a2a(prev_att_v)
            prev_att_v = F.dropout(prev_att_v)
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            prev_att_h = F.dropout(prev_att_h)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = torch.add(prev_att_v, prev_att_h_1)
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                # ##################################################
                # lstm core
                # batch_size * 4*rnn_size
                all_input_sums = self.i2hs[j](xt) + self.h2hs[j](prev_h) + self.a2hs[j](prev_att_res)

                # batch_size * 4*rnn_size
                sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)

                in_gate = sigmoid_chunk[:,0:self.rnn_size]
                forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
                out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

                in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
                in_transform = F.tanh(in_transform)

                next_c = (forget_gate * prev_c) + (in_gate * in_transform)

                # batch_size * rnn_size
                next_h = out_gate * F.tanh(next_c)
                # ##################################################

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()

            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * input_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            att_v = F.dropout(att_v)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            att_h = F.dropout(att_h)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
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

            outputs.append(next_c)
            outputs.append(top_h)

        return outputs

class LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_SET(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, rnn_size_list, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_SET, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_parallels = num_parallels
        self.rnn_size_list = rnn_size_list
        self.att_size = att_size
        self.dropout = dropout
        self.rnn_size = rnn_size

        self.lstms = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.x_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()

        for size in self.rnn_size_list:
            lstm = LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_ITEM(input_size, num_layers, num_parallels, size, att_size, dropout)
            self.lstms.append(lstm)

            linear = nn.Linear(size, self.rnn_size)
            init.xavier_normal(linear.weight)
            self.linears.append(linear)

            x_linear = nn.Linear(self.input_size, size)
            init.xavier_normal(x_linear.weight)
            self.x_linears.append(x_linear)

            a_linear = nn.Linear(self.input_size, size)
            init.xavier_normal(a_linear.weight)
            self.a_linears.append(a_linear)

        self.proj = nn.Linear(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * input_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        top_hs = []
        total_outputs = []
        for i in range(len(self.rnn_size_list)):

            size = self.rnn_size_list[i]

            x1 = self.x_linears[i](x)
            att1 = self.a_linears[i](att.view(-1,self.input_size)).view(-1,self.att_size,size)

            outputs = self.lstms[i](x1, att1, inputs[i*2*self.num_layers: (i+1)*2*self.num_layers])
            total_outputs.extend(outputs)

            top_h = outputs[-1]
            top_h = self.linears[i](top_h)
            top_hs.append(top_h)

        top_h = torch.cat([_.unsqueeze(1) for _ in top_hs], 1).mean(1).squeeze()

        # policy
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        return total_outputs, logsoft


class LSTM_DOUBLE_ATT_STACK_PARALLEL_BN(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_BN, self).__init__()

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

        for i in range(self.num_parallels):

            a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(a2h.weight)
            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)

            self.a2hs.append(a2h)
            self.i2hs.append(i2h)
            self.h2hs.append(h2h)


        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        self.proj = nn.Linear(self.rnn_size, self.output_size)

        # batch_norm


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
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

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
            prev_att_v = F.relu(prev_att_v)
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            prev_att_h = F.relu(prev_att_h)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = torch.add(prev_att_v, prev_att_h_1)
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = F.relu(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                # ##################################################
                # lstm core
                # batch_size * 4*rnn_size
                all_input_sums = self.i2hs[j](xt) + self.h2hs[j](prev_h) + self.a2hs[j](prev_att_res)

                # batch_size * 4*rnn_size
                sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)

                in_gate = sigmoid_chunk[:,0:self.rnn_size]
                forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
                out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

                in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
                in_transform = F.tanh(in_transform)

                next_c = (forget_gate * prev_c) + (in_gate * in_transform)

                # batch_size * rnn_size
                next_h = out_gate * F.tanh(next_c)
                # ##################################################

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()

            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            att_v = F.relu(att_v)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            att_h = F.relu(att_h)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
            dot = F.relu(dot)
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

            outputs.append(next_c)
            outputs.append(top_h)

        # policy
        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft


class LSTM_DOUBLE_ATT_STACK_PARALLEL_BN_RELU(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_BN_RELU, self).__init__()

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

        for i in range(self.num_parallels):

            a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(a2h.weight)
            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)

            self.a2hs.append(a2h)
            self.i2hs.append(i2h)
            self.h2hs.append(h2h)


        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        self.proj = nn.Linear(self.rnn_size, self.output_size)

        # batch_norm
        self.bn_h2a = nn.BatchNorm1d(self.att_size)
        self.bn_h2a1 = nn.BatchNorm1d(self.att_size)

        self.bn_res = nn.BatchNorm1d(self.rnn_size)
        self.bn_res1 = nn.BatchNorm1d(self.rnn_size)

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
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

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
            prev_att_v = F.relu(prev_att_v)
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            prev_att_h = self.bn_h2a(prev_att_h)
            prev_att_h = F.relu(prev_att_h)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = torch.add(prev_att_v, prev_att_h_1)
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            prev_att_res = self.bn_res(prev_att_res)
            prev_att_res = F.relu(prev_att_res)
            # spatial attention end
            # ##################################################

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                # ##################################################
                # lstm core
                # batch_size * 4*rnn_size
                all_input_sums = self.i2hs[j](xt) + self.h2hs[j](prev_h) + self.a2hs[j](prev_att_res)

                # batch_size * 4*rnn_size
                sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)

                in_gate = sigmoid_chunk[:,0:self.rnn_size]
                forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
                out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

                in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
                in_transform = F.tanh(in_transform)

                next_c = (forget_gate * prev_c) + (in_gate * in_transform)

                # batch_size * rnn_size
                next_h = out_gate * F.tanh(next_c)
                # ##################################################

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()

            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            att_v = F.relu(att_v)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            att_h = self.bn_h2a1(att_h)
            att_h = F.relu(att_h)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
            dot = dot.view(-1, self.att_size)

            weight = F.softmax(dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1, 2)
            # batch_size * rnn_size
            att_res = torch.bmm(att_seq_t, weight.unsqueeze(2)).squeeze()
            att_res = self.bn_res1(att_res)
            att_res = F.relu(att_res)
            # spatial attention end
            # ##################################################

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)
            outputs.append(top_h)

        # policy
        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft

class LSTM_DOUBLE_ATT_STACK_PARALLEL_POLICY(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_POLICY, self).__init__()

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

        for i in range(self.num_parallels):

            a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(a2h.weight)
            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)

            self.a2hs.append(a2h)
            self.i2hs.append(i2h)
            self.h2hs.append(h2h)


        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        self.policy = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.rnn_size*2, self.rnn_size*4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.rnn_size*4, self.output_size)
        )

        # self.bn_h = nn.BatchNorm1d(self.rnn_size)
        # self.bn_a = nn.BatchNorm1d(self.rnn_size)

        # self.bn_h1 = nn.BatchNorm1d(self.rnn_size)
        self.bn_a1 = nn.BatchNorm1d(self.rnn_size)


        # init parameters
        init.xavier_normal(self.a2a.weight)
        init.xavier_normal(self.h2a.weight)
        init.xavier_normal(self.d2d.weight)

        init.xavier_normal(self.a2a1.weight)
        init.xavier_normal(self.h2a1.weight)
        init.xavier_normal(self.d2d1.weight)

        init.xavier_normal(self.policy[0].weight)
        init.xavier_normal(self.policy[3].weight)
        init.xavier_normal(self.policy[6].weight)

        # batch_size * rnn_size


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

            # ##################################################
            # spatial attention start
            # (batch * att_size) * rnn_size
            prev_att_v = att.view(-1, self.rnn_size)
            # (batch * att_size) * att_size
            prev_att_v = self.a2a(prev_att_v)
            # batch * att_size * att_size
            prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            prev_att_h = self.h2a(prev_h)
            # batch * att_size * att_size
            prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

            # batch * att_size * att_size
            prev_dot = prev_att_v + prev_att_h_1
            prev_dot = F.tanh(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)
            prev_dot = self.d2d(prev_dot)
            prev_dot = prev_dot.view(-1, self.att_size)

            # batch_size * att_size
            prev_weight = F.softmax(prev_dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1,2)
            # batch_size * rnn_size
            prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                # ##################################################
                # lstm core
                # batch_size * 4*rnn_size
                all_input_sums = self.i2hs[j](xt) + self.h2hs[j](prev_h) + self.a2hs[j](prev_att_res)

                # batch_size * 4*rnn_size
                sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)

                in_gate = sigmoid_chunk[:,0:self.rnn_size]
                forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
                out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

                in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
                in_transform = F.tanh(in_transform)

                next_c = (forget_gate * prev_c) + (in_gate * in_transform)

                # batch_size * rnn_size
                next_h = out_gate * F.tanh(next_c)
                # ##################################################

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()

            # dropout
            # next_h = F.dropout(next_h)

            # ##################################################
            # spatial attention start
            # (batch_size * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            att_v = self.a2a1(att_v)
            # batch_size * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)

            # batch_size * att_size
            att_h = self.h2a1(next_h)
            # relu
            # att_h = F.relu(att_h)
            # batch_size * att_size * att_size
            att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

            # Batch_Normalize
            # att_v = self.bn_a(att_v)
            # att_h_1 = self.bn_h(att_h_1)

            # batch_size * att_size * att_size
            dot = att_h_1 + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2d1(dot)
            dot = dot.view(-1, self.att_size)

            weight = F.softmax(dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1, 2)
            # batch_size * rnn_size
            att_res = torch.bmm(att_seq_t, weight.unsqueeze(2)).squeeze()
            # spatial attention end
            # ##################################################

            # Batch_Normalize
            att_res = self.bn_a1(att_res)
            # next_h = self.bn_h1(next_h)

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)
            outputs.append(top_h)

        # policy
        top_h = outputs[-1]
        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.policy(top_h))

        return outputs, logsoft

class LSTM_DOUBLE_ATT_TOP(nn.Module):

    def __init__(self, input_size, output_size, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_TOP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout

        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        # init parameters
        init.xavier_normal(self.a2a.weight)
        init.xavier_normal(self.h2a.weight)
        init.xavier_normal(self.d2d.weight)

        init.xavier_normal(self.a2h.weight)
        init.xavier_normal(self.i2h.weight)
        init.xavier_normal(self.h2h.weight)

        init.xavier_normal(self.a2a1.weight)
        init.xavier_normal(self.h2a1.weight)
        init.xavier_normal(self.d2d1.weight)

        # batch_size * rnn_size


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, prev_h, prev_c):

        # ##################################################
        # spatial attention start
        # (batch * att_size) * rnn_size
        prev_att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * att_size
        prev_att_v = self.a2a(prev_att_v)
        # batch * att_size * att_size
        prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
        # batch * att_size
        prev_att_h = self.h2a(prev_h)
        # batch * att_size * att_size
        prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

        # batch * att_size * att_size
        prev_dot = torch.add(prev_att_v, prev_att_h_1)
        prev_dot = F.tanh(prev_dot)
        prev_dot = prev_dot.view(-1, self.att_size)
        prev_dot = self.d2d(prev_dot)
        prev_dot = prev_dot.view(-1, self.att_size)

        # batch_size * att_size
        prev_weight = F.softmax(prev_dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1,2)
        # batch_size * rnn_size
        prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
        # spatial attention end
        # ##################################################


        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(x) + self.h2h(prev_h) + self.a2h(prev_att_res)

        # batch_size * 4*rnn_size
        sigmoid_chunk = all_input_sums[:,:3*self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:,0:self.rnn_size]
        forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
        out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

        in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)

        # batch_size * rnn_size
        next_h = out_gate * F.tanh(next_c)
        # ##################################################


        # ##################################################
        # spatial attention start
        # (batch_size * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        att_v = self.a2a1(att_v)
        # batch_size * att_size * att_size
        att_v = att_v.view(-1, self.att_size, self.att_size)

        # batch_size * att_size
        att_h = self.h2a1(next_h)
        # batch_size * att_size * att_size
        att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

        # batch_size * att_size * att_size
        dot = att_h_1 + att_v
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_size)
        dot = self.d2d1(dot)
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

        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)

        outputs = []
        outputs.append(next_c)
        outputs.append(next_h)

        return outputs, top_h


class LSTM_DOUBLE_ATT_RELU(nn.Module):

    def __init__(self, input_size, output_size, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_RELU, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout

        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

        # attention
        self.a2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.h2a1 = nn.Linear(self.rnn_size, self.att_size)
        self.d2d1 = nn.Linear(self.att_size, 1)

        self.proj = nn.Linear(self.rnn_size, self.output_size)

        self.relu = nn.RReLU(inplace=True)

        # init parameters
        init.xavier_normal(self.a2a.weight)
        init.xavier_normal(self.h2a.weight)
        init.xavier_normal(self.d2d.weight)

        init.xavier_normal(self.i2h.weight)
        init.xavier_normal(self.a2h.weight)
        init.xavier_normal(self.h2h.weight)

        init.xavier_normal(self.a2a1.weight)
        init.xavier_normal(self.h2a1.weight)
        init.xavier_normal(self.d2d1.weight)

        init.xavier_normal(self.proj.weight)

        # batch_size * rnn_size

    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, prev_h, prev_c):
        # ##################################################
        # spatial attention start
        # (batch * att_size) * rnn_size
        prev_att_v = att.view(-1, self.rnn_size)
        # (batch * att_size) * att_size
        prev_att_v = self.a2a(prev_att_v)
        # batch * att_size * att_size
        prev_att_v = prev_att_v.view(-1, self.att_size, self.att_size)
        # batch * att_size
        prev_att_h = self.h2a(prev_h)
        # batch * att_size * att_size
        prev_att_h_1 = prev_att_h.unsqueeze(2).expand_as(prev_att_v)

        # batch * att_size * att_size
        prev_dot = torch.add(prev_att_v, prev_att_h_1)
        prev_dot = self.relu(prev_dot)
        prev_dot = prev_dot.view(-1, self.att_size)
        prev_dot = self.d2d(prev_dot)
        prev_dot = prev_dot.view(-1, self.att_size)

        # batch_size * att_size
        prev_weight = F.softmax(prev_dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1, 2)
        # batch_size * rnn_size
        prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
        # spatial attention end
        # ##################################################


        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(x) + self.h2h(prev_h) + self.a2h(prev_att_res)

        # batch_size * 4*rnn_size
        sigmoid_chunk = all_input_sums[:, :3 * self.rnn_size]
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk[:, 0:self.rnn_size]
        forget_gate = sigmoid_chunk[:, self.rnn_size:self.rnn_size * 2]
        out_gate = sigmoid_chunk[:, self.rnn_size * 2:self.rnn_size * 3]

        in_transform = all_input_sums[:, self.rnn_size * 3:self.rnn_size * 4]
        in_transform = F.tanh(in_transform)

        next_c = (forget_gate * prev_c) + (in_gate * in_transform)

        # batch_size * rnn_size
        next_h = out_gate * F.tanh(next_c)
        # ##################################################


        # ##################################################
        # spatial attention start
        # (batch_size * att_size) * rnn_size
        att_v = att.view(-1, self.rnn_size)
        att_v = self.a2a1(att_v)
        # batch_size * att_size * att_size
        att_v = att_v.view(-1, self.att_size, self.att_size)

        # batch_size * att_size
        att_h = self.h2a1(next_h)
        # batch_size * att_size * att_size
        att_h_1 = att_h.unsqueeze(2).expand_as(att_v)

        # batch_size * att_size * att_size
        dot = att_h_1 + att_v
        dot = self.relu(dot)
        dot = dot.view(-1, self.att_size)
        dot = self.d2d1(dot)
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

        if self.dropout > 0:
            top_h = F.dropout(top_h, self.dropout)
        logsoft = F.log_softmax(self.proj(top_h))

        outputs = []
        outputs.append(next_c)
        outputs.append(next_h)

        return outputs, logsoft


