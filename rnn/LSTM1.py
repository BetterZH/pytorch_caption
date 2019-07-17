import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import torch.nn.init as init
import math


# Input: :math:`(N, C_{in}, L_{in})`
# Output: :math:`(N, C_{out}, L_{out})`
class lstm_block(nn.Module):

    def __init__(self):
        super(lstm_block, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(3)

        self.conv2 = nn.Conv1d(3, 3, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(3)

        self.conv3 = nn.Conv1d(3, 1, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm1d(1)

        self.relu = nn.ReLU()

        init.xavier_normal(self.conv1.weight)
        init.xavier_normal(self.conv2.weight)
        init.xavier_normal(self.conv3.weight)


    # batch_size * rnn_size
    def forward(self, x):

        # batch_size * 1 * rnn_size
        x = x.unsqueeze(1)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out += residual
        out = self.relu(out)

        # batch_size * rnn_size
        out = out.squeeze()

        # batch_size * rnn_size
        return out

class lstm_conv(nn.Module):

    def __init__(self):
        super(lstm_conv, self).__init__()
        self.layer = self._make_layer(lstm_block(), 4)

    def _make_layer(self, block, block_num):

        layers = []
        for i in range(block_num):
            layers.append(block)

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.layer(x)

        return x



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



class CONV_LSTM(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, dropout, num_layers, block_num, use_proj_mul):
        super(CONV_LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.num_layers = num_layers

        # core
        self.convs = nn.ModuleList()

        for i in range(self.num_layers * 8):
            # block = nn.Linear(self.rnn_size, self.rnn_size)
            block = lstm_linear(self.rnn_size, block_num)
            self.convs.append(block)

        if use_proj_mul:
            self.proj = nn.Sequential(
                # rnn_size   512 -> 2048
                nn.Linear(self.rnn_size, self.rnn_size*4),
                nn.ReLU(),
                nn.Dropout(),
                # rnn_size   2048 -> 4096
                nn.Linear(self.rnn_size*4, self.rnn_size*8),
                nn.ReLU(),
                nn.Dropout(),
                # rnn_size -> output_size  4096 -> 9982
                nn.Linear(self.rnn_size*8, self.output_size)
            )
        else:
            self.proj = nn.Linear(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, inputs):

        outputs = []

        for i in range(self.num_layers):

            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            ############ core ############
            # batch_size * 2*rnn_size
            input_gate = F.sigmoid(self.convs[i*8 + 0](x) + self.convs[i*8 + 1](prev_h))
            forget_gate = F.sigmoid(self.convs[i*8 + 2](x) + self.convs[i*8 + 3](prev_h))
            output_gate = F.sigmoid(self.convs[i*8 + 4](x) + self.convs[i*8 + 5](prev_h))
            in_transform = F.tanh(self.convs[i*8 + 6](x) + self.convs[i*8 + 7](prev_h))

            next_c = forget_gate * prev_c + input_gate * in_transform
            next_h = output_gate * F.tanh(next_c)
            ############ core ############


            if self.dropout > 0:
                next_h = F.dropout(next_h, self.dropout)

            outputs.append(next_c)
            outputs.append(next_h)

        logsolft = F.log_softmax(self.proj(outputs[-1]))

        # next_h   : batch_size * rnn_size
        # next_c   : batch_size * rnn_size
        # logsofts : batch_size * (vocab_size + 1)
        return outputs, logsolft


class CONV_IT_ATT_COMBINE(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, att_size, dropout, num_layers, word_input_layer, att_input_layer):
        super(CONV_IT_ATT_COMBINE, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.word_input_layer = word_input_layer
        self.att_input_layer = att_input_layer

        # input combine
        self.w2ic = lstm_conv()
        self.i2ic = lstm_conv()

        # atten combine
        self.a2ac = lstm_conv()
        self.h2ac = lstm_conv()

        # word combine
        self.w2wc = lstm_conv()
        self.h2wc = lstm_conv()


        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.i2his = nn.ModuleList()
        self.h2his = nn.ModuleList()

        self.i2hfs = nn.ModuleList()
        self.h2hfs = nn.ModuleList()

        self.i2hos = nn.ModuleList()
        self.h2hos = nn.ModuleList()

        self.i2hgs = nn.ModuleList()
        self.h2hgs = nn.ModuleList()

        for i in range(self.num_layers):

            self.i2his.append(lstm_conv())
            self.h2his.append(lstm_conv())

            self.i2hfs.append(lstm_conv())
            self.h2hfs.append(lstm_conv())

            self.i2hos.append(lstm_conv())
            self.h2hos.append(lstm_conv())

            self.i2hgs.append(lstm_conv())
            self.h2hgs.append(lstm_conv())


        self.proj = nn.Linear(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, w, att, inputs):

        outputs = []
        for i in range(self.num_layers):

            prev_c = inputs[i*2]
            prev_h = inputs[1*2+1]

            if i == 0:
                xt = F.tanh(self.i2ic(x) + self.w2ic(w))
            elif i == self.word_input_layer:
                xt = F.tanh(self.w2wc(w) + self.h2wc(outputs[-1]))
            else:
                xt = outputs[-1]

            if i == self.att_input_layer:
                ############ attention ############
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
                ############ attention ############

                xt = F.tanh(self.h2ac(xt) + self.a2ac(att_res))


            ############ core ############
            # batch_size * 2*rnn_size
            input_gate = F.sigmoid(self.i2his[i](xt) + self.h2his[i](prev_h))
            forget_gate = F.sigmoid(self.i2hfs[i](xt) + self.h2hfs[i](prev_h))
            output_gate = F.sigmoid(self.i2hos[i](xt) + self.h2hos[i](prev_h))
            in_transform = F.tanh(self.i2hgs[i](xt) + self.h2hgs[i](prev_h))

            next_c = forget_gate * prev_c + input_gate * in_transform
            next_h = output_gate * F.tanh(next_c)
            ############ core ############


            if self.dropout > 0:
                next_h = F.dropout(next_h, self.dropout)

            outputs.append(next_c)
            outputs.append(next_h)

        top_h = outputs[-1]
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft


class FO_IT_ATT_COMBINE(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, att_size, dropout, num_layers, word_input_layer, att_input_layer):
        super(FO_IT_ATT_COMBINE, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.word_input_layer = word_input_layer
        self.att_input_layer = att_input_layer

        # input combine
        self.w2ic = nn.Linear(self.input_size, self.rnn_size)
        self.i2ic = nn.Linear(self.input_size, self.rnn_size)

        # atten combine
        self.a2ac = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2ac = nn.Linear(self.rnn_size, self.rnn_size)

        # word combine
        self.w2wc = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2wc = nn.Linear(self.rnn_size, self.rnn_size)


        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.i2hs = nn.ModuleList()
        self.h2hs = nn.ModuleList()
        # self.r2as = nn.ModuleList()

        for i in range(self.num_layers):

            if i == 0:
                i2h = nn.Linear(self.input_size, 2 * self.rnn_size)
            else:
                i2h = nn.Linear(self.rnn_size, 2 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 2 * self.rnn_size)
            # r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)
            # init.xavier_normal(r2a.weight)

            self.i2hs.append(i2h)
            self.h2hs.append(h2h)
            # self.r2as.append(r2a)

        self.proj = nn.Linear(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, w, att, inputs):

        outputs = []
        for i in range(self.num_layers):

            prev_c = inputs[i*2]
            prev_h = inputs[1*2+1]

            if i == 0:
                xt = F.tanh(self.i2ic(x) + self.w2ic(w))
            elif i == self.word_input_layer:
                xt = F.tanh(self.w2wc(w) + self.h2wc(outputs[-1]))
            else:
                xt = outputs[-1]

            if i == self.att_input_layer:
                ############ attention ############
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
                ############ attention ############

                xt = F.tanh(self.h2ac(xt) + self.a2ac(att_res))


            ############ core ############
            # batch_size * 2*rnn_size
            all_input_sums = self.i2hs[i](xt) + self.h2hs[i](prev_h)
            sigmoid_chunk = F.sigmoid(all_input_sums)
            forget_gate = sigmoid_chunk[:,0:self.rnn_size]
            out_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]

            next_c = forget_gate * prev_c
            next_h = out_gate * F.tanh(next_c)
            ############ core ############


            if self.dropout > 0:
                next_h = F.dropout(next_h, self.dropout)

            outputs.append(next_c)
            outputs.append(next_h)

        top_h = outputs[-1]
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft

class LSTM_IT_ATT_COMBINE(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, att_size, dropout, num_layers, word_input_layer, att_input_layer):
        super(LSTM_IT_ATT_COMBINE, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.word_input_layer = word_input_layer
        self.att_input_layer = att_input_layer

        # input combine
        self.w2ic = nn.Linear(self.input_size, self.rnn_size)
        self.i2ic = nn.Linear(self.input_size, self.rnn_size)

        # atten combine
        self.a2ac = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2ac = nn.Linear(self.rnn_size, self.rnn_size)

        # word combine
        self.w2wc = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2wc = nn.Linear(self.rnn_size, self.rnn_size)


        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.i2hs = nn.ModuleList()
        self.h2hs = nn.ModuleList()
        # self.r2as = nn.ModuleList()

        for i in range(self.num_layers):

            if i == 0:
                i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            else:
                i2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            # r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)
            # init.xavier_normal(r2a.weight)

            self.i2hs.append(i2h)
            self.h2hs.append(h2h)
            # self.r2as.append(r2a)

        self.proj = nn.Linear(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, w, att, inputs):

        outputs = []
        for i in range(self.num_layers):

            prev_c = inputs[i*2]
            prev_h = inputs[1*2+1]

            if i == 0:
                xt = F.tanh(self.i2ic(x) + self.w2ic(w))
            elif i == self.word_input_layer:
                xt = F.tanh(self.w2wc(w) + self.h2wc(outputs[-1]))
            else:
                xt = outputs[-1]

            if i == self.att_input_layer:
                ############ attention ############
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
                ############ attention ############

                xt = F.tanh(self.h2ac(xt) + self.a2ac(att_res))


            ############ core ############
            # batch_size * 4*rnn_size
            all_input_sums = self.i2hs[i](xt) + self.h2hs[i](prev_h)

            sigmoid_chunk = all_input_sums[:,0:3*self.rnn_size]
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)

            in_gate = sigmoid_chunk[:,0:self.rnn_size]
            forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
            out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

            in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
            in_transform = F.tanh(in_transform)

            next_c = (forget_gate * prev_c) + (in_gate * in_transform)
            next_h = out_gate * F.tanh(next_c)
            ############ core ############


            if self.dropout > 0:
                next_h = F.dropout(next_h, self.dropout)

            outputs.append(next_c)
            outputs.append(next_h)

        top_h = outputs[-1]
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft


class LSTM_IT_ATT(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, att_size, dropout, num_layers, word_input_layer, att_input_layer):
        super(LSTM_IT_ATT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.word_input_layer = word_input_layer
        self.att_input_layer = att_input_layer

        # attention
        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

        # core
        self.i2hs = nn.ModuleList()
        self.h2hs = nn.ModuleList()
        # self.r2as = nn.ModuleList()

        for i in range(self.num_layers):

            if i == 0:
                i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            else:
                i2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            # r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)
            # init.xavier_normal(r2a.weight)

            self.i2hs.append(i2h)
            self.h2hs.append(h2h)
            # self.r2as.append(r2a)

        self.proj = nn.Linear(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, w, att, inputs):

        outputs = []
        for i in range(self.num_layers):

            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            if i == 0:
                xt = x + w
            elif i == self.word_input_layer:
                xt = outputs[-1] + w
            else:
                xt = outputs[-1]

            if i == self.att_input_layer:
                ############ attention ############
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
                ############ attention ############

                xt = xt + att_res


            ############ core ############
            # batch_size * 4*rnn_size
            all_input_sums = self.i2hs[i](xt) + self.h2hs[i](prev_h)

            sigmoid_chunk = all_input_sums[:,0:3*self.rnn_size]
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)

            in_gate = sigmoid_chunk[:,0:self.rnn_size]
            forget_gate = sigmoid_chunk[:,self.rnn_size:self.rnn_size*2]
            out_gate = sigmoid_chunk[:,self.rnn_size*2:self.rnn_size*3]

            in_transform = all_input_sums[:,self.rnn_size*3:self.rnn_size*4]
            in_transform = F.tanh(in_transform)

            next_c = (forget_gate * prev_c) + (in_gate * in_transform)
            next_h = out_gate * F.tanh(next_c)
            ############ core ############


            if self.dropout > 0:
                next_h = F.dropout(next_h, self.dropout)

            outputs.append(next_c)
            outputs.append(next_h)

        top_h = outputs[-1]
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft



class LSTM_SOFT_ATT(nn.Module):
    def __init__(self, input_size, output_size, rnn_size, att_size, dropout, num_layers, word_input_layer):
        super(LSTM_SOFT_ATT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_size = att_size
        self.num_layers = num_layers
        self.word_input_layer = word_input_layer

        # core
        self.a2as = nn.ModuleList()
        self.h2as = nn.ModuleList()
        self.d2ds = nn.ModuleList()

        self.i2hs = nn.ModuleList()
        self.h2hs = nn.ModuleList()
        self.r2as = nn.ModuleList()

        for i in range(self.num_layers):

            a2a = nn.Linear(self.rnn_size, self.att_size)
            h2a = nn.Linear(self.rnn_size, self.att_size)
            d2d = nn.Linear(self.att_size, 1)

            i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
            h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            r2a = nn.Linear(self.rnn_size, 4 * self.rnn_size)

            init.xavier_normal(a2a.weight)
            init.xavier_normal(h2a.weight)
            init.xavier_normal(d2d.weight)

            init.xavier_normal(i2h.weight)
            init.xavier_normal(h2h.weight)
            init.xavier_normal(r2a.weight)

            self.a2as.append(a2a)
            self.h2as.append(h2a)
            self.d2ds.append(d2d)

            self.i2hs.append(i2h)
            self.h2hs.append(h2h)
            self.r2as.append(r2a)



    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, w, att, inputs):

        outputs = []
        for i in range(self.num_layers):

            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            if i == 0:
                xt = x
            elif i == self.word_input_layer:
                xt = w + outputs[-1]
                # xt = torch.cat([w, outputs[-1]], 1)
            else:
                xt = outputs[-1]

            # (batch * att_size) * rnn_size
            att_v = att.view(-1, self.rnn_size)
            # (batch * att_size) * att_size
            att_v = self.a2as[i](att_v)
            # batch * att_size * att_size
            att_v = att_v.view(-1, self.att_size, self.att_size)
            # batch * att_size
            att_h = self.h2as[i](prev_h)
            # batch * att_size * att_size
            att_h = att_h.unsqueeze(2).expand_as(att_v)

            # batch * att_size * att_size
            dot = att_h + att_v
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_size)
            dot = self.d2ds[i](dot)
            dot = dot.view(-1, self.att_size)

            # batch_size * att_size
            weigth = F.softmax(dot)
            # batch_size * rnn_size * att_size
            att_seq_t = att.transpose(1, 2)
            # batch_size * rnn_size
            att_res = torch.bmm(att_seq_t, weigth.unsqueeze(2)).squeeze()

            # batch_size * 4*rnn_size
            all_input_sums = self.i2hs[i](xt) + self.h2hs[i](prev_h) + self.r2as[i](att_res)

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

            outputs.append(next_c)
            outputs.append(next_h)


        return outputs

class lstm_core_with_att(nn.Module):
    def __init__(self, input_size, rnn_size):
        super(lstm_core_with_att, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

    def forward(self, xt, prev_c, prev_h, prev_att_res):
        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(xt) + self.h2h(prev_h) + self.a2h(prev_att_res)

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


class lstm_soft_att(nn.Module):

    def __init__(self, rnn_size, att_size):
        super(lstm_soft_att, self).__init__()

        self.rnn_size = rnn_size
        self.att_size = att_size

        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

    def forward(self, att, prev_h):

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
        # batch * att_size * att_size
        prev_dot = F.tanh(prev_dot)
        # (batch * att_size) * att_size
        prev_dot = prev_dot.view(-1, self.att_size)
        # (batch * att_size) * 1
        prev_dot = self.d2d(prev_dot)
        # batch * att_size
        prev_dot = prev_dot.view(-1, self.att_size)

        # batch_size * att_size
        prev_weight = F.softmax(prev_dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1, 2)

        # batch_size * rnn_size
        prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
        # spatial attention end
        # ##################################################

        return prev_att_res


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

        # core
        self.cores = nn.ModuleList()
        for i in range(self.num_layers * self.num_parallels):
            core = lstm_core_with_att(self.input_size, self.rnn_size)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers * 2):
            att = lstm_soft_att(self.rnn_size, self.att_size)
            self.attens.append(att)

        # proj
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

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i*2+0](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):

                next_c, next_h = self.cores[i*self.num_parallels+j](xt, prev_c, prev_h, prev_att_res)

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            # next_c = next_c.max(1)[0].squeeze()

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()
            # next_h = next_h.max(1)[0].squeeze()

            att_res = self.attens[i*2+1](att, next_h)

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

        # policy
        top_h = outputs[-1]
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft

# with n num_layers
class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT, self).__init__()

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
            core = lstm_core_with_att(self.input_size, self.rnn_size)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers * 2):
            att = lstm_soft_att(self.rnn_size, self.att_size)
            self.attens.append(att)

        # proj
        self.projs = nn.ModuleList()
        for i in range(self.num_layers):
            proj = nn.Linear(self.rnn_size, self.output_size)
            self.projs.append(proj)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        logprobs = []
        for i in range(self.num_layers):
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i*2+0](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):

                next_c, next_h = self.cores[i*self.num_parallels+j](xt, prev_c, prev_h, prev_att_res)

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            # next_c = next_c.max(1)[0].squeeze()

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()
            # next_h = next_h.max(1)[0].squeeze()

            att_res = self.attens[i*2+1](att, next_h)

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

            # policy
            logsoft = F.log_softmax(self.projs[i](top_h))

            logprobs.append(logsoft)


        return outputs, logprobs


# one proj
class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_NEW(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_NEW, self).__init__()

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
            core = lstm_core_with_att(self.input_size, self.rnn_size)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers * 2):
            att = lstm_soft_att(self.rnn_size, self.att_size)
            self.attens.append(att)

        # proj
        self.proj = nn.Linear(self.rnn_size, self.output_size)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        logprobs = []
        for i in range(self.num_layers):
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i*2+0](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):

                next_c, next_h = self.cores[i*self.num_parallels+j](xt, prev_c, prev_h, prev_att_res)

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            # next_c = next_c.max(1)[0].squeeze()

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()
            # next_h = next_h.max(1)[0].squeeze()

            att_res = self.attens[i*2+1](att, next_h)

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

            # policy
            logsoft = F.log_softmax(self.proj(top_h))

            logprobs.append(logsoft)


        return outputs, logprobs


class lstm_core_with_att_mul_in(nn.Module):
    def __init__(self, input_size, rnn_size):
        super(lstm_core_with_att_mul_in, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size

        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.a2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

    def forward(self, xt, prev_c, prev_h, prev_att_res):
        # ##################################################
        # lstm core
        # batch_size * 4*rnn_size
        all_input_sums = self.i2h(xt) + self.h2h(prev_h) + self.a2h(prev_att_res)

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


class lstm_soft_att(nn.Module):

    def __init__(self, rnn_size, att_size):
        super(lstm_soft_att, self).__init__()

        self.rnn_size = rnn_size
        self.att_size = att_size

        self.a2a = nn.Linear(self.rnn_size, self.att_size)
        self.h2a = nn.Linear(self.rnn_size, self.att_size)
        self.d2d = nn.Linear(self.att_size, 1)

    def forward(self, att, prev_h):

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
        # batch * att_size * att_size
        prev_dot = F.tanh(prev_dot)
        # (batch * att_size) * att_size
        prev_dot = prev_dot.view(-1, self.att_size)
        # (batch * att_size) * 1
        prev_dot = self.d2d(prev_dot)
        # batch * att_size
        prev_dot = prev_dot.view(-1, self.att_size)

        # batch_size * att_size
        prev_weight = F.softmax(prev_dot)
        # batch_size * rnn_size * att_size
        att_seq_t = att.transpose(1, 2)

        # batch_size * rnn_size
        prev_att_res = torch.bmm(att_seq_t, prev_weight.unsqueeze(2)).squeeze()
        # spatial attention end
        # ##################################################

        return prev_att_res

class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_MUL_IN(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_MUL_IN, self).__init__()

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
            core = lstm_core_with_att(self.input_size, self.rnn_size)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers * 2):
            att = lstm_soft_att(self.rnn_size, self.att_size)
            self.attens.append(att)

        # proj
        self.projs = nn.ModuleList()
        for i in range(self.num_parallels):
            proj = nn.Linear(self.rnn_size, self.output_size)
            self.projs.append(proj)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs, inputs_1):

        outputs = []
        logprobs = []
        for i in range(self.num_layers):
            prev_c = inputs[i*2]
            prev_h = inputs[i*2+1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i*2+0](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):

                next_c, next_h = self.cores[i*self.num_parallels+j](xt, prev_c, prev_h, prev_att_res)

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1).mean(1).squeeze()
            # next_c = next_c.max(1)[0].squeeze()

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1).mean(1).squeeze()
            # next_h = next_h.max(1)[0].squeeze()

            att_res = self.attens[i*2+1](att, next_h)

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

            # policy
            logsoft = F.log_softmax(self.projs[i](top_h))

            logprobs.append(logsoft)


        return outputs, logprobs

