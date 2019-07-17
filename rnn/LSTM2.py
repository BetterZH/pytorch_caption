import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import *
import torch.nn.init as init
import math
import CORE
import ATT


# with n num_layers
# with multi proj
class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_WEIGHT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_WEIGHT, self).__init__()

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
        for i in range(self.num_layers * 2):
            att = ATT.lstm_att_with_x_att_h(self.rnn_size, self.att_size)
            self.attens.append(att)

        # proj
        self.projs = nn.ModuleList()
        for i in range(self.num_layers):
            proj = nn.Linear(self.rnn_size, self.output_size)
            self.projs.append(proj)

        # proj weight
        self.proj_weight = nn.Linear(self.rnn_size, self.output_size)

        init.xavier_normal(self.proj_weight.weight)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        proj_w = F.sigmoid(self.proj_weight(x))

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

            prev_att_res = self.attens[i*2+0](xt, att, prev_h)

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

            att_res = self.attens[i*2+1](xt, att, next_h)

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

            # policy
            logsoft = F.log_softmax(self.projs[i](top_h) * proj_w)

            logprobs.append(logsoft)


        return outputs, logprobs, proj_w

# with n num_layers
# new
class LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT, self).__init__()

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

        # proj weight
        self.proj_weight = nn.Linear(self.rnn_size, self.output_size)

        init.xavier_normal(self.proj_weight.weight)


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

        proj_w = F.sigmoid(self.proj_weight(x))

        logsoft = F.log_softmax(self.proj(top_h) * proj_w)


        return outputs, logsoft, proj_w

# with n num_layers
# new
class LSTM_SOFT_ATT_STACK_PARALLEL_WITH_FC_WEIGHT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT_STACK_PARALLEL_WITH_FC_WEIGHT, self).__init__()

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

        # proj weight
        self.proj_weight = nn.Linear(self.rnn_size, self.output_size)

        init.xavier_normal(self.proj.weight)
        init.xavier_normal(self.proj_weight.weight)

    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        for i in range(self.num_layers):
            prev_c = inputs[i * 2]
            prev_h = inputs[i * 2 + 1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                next_c, next_h = self.cores[i * self.num_parallels + j](xt, prev_c, prev_h, prev_att_res)

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

        fc_feats = att.mean(1).view_as(x)

        proj_w = F.sigmoid(self.proj_weight(fc_feats + x))

        logsoft = F.log_softmax(self.proj(top_h) * proj_w)

        return outputs, logsoft, proj_w


# with n num_layers
# new
class LSTM_SOFT_ATT_STACK_PARALLEL_WITH_MUL_WEIGHT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_SOFT_ATT_STACK_PARALLEL_WITH_MUL_WEIGHT, self).__init__()

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

        # proj weight
        linear_weight_1 = nn.Linear(self.rnn_size, self.rnn_size)
        linear_weight_2 = nn.Linear(self.rnn_size, self.output_size)

        init.xavier_normal(linear_weight_1.weight)
        init.xavier_normal(linear_weight_2.weight)

        self.proj_weight = nn.Sequential(linear_weight_1,
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         linear_weight_2)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, inputs):

        outputs = []
        for i in range(self.num_layers):
            prev_c = inputs[i * 2]
            prev_h = inputs[i * 2 + 1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i](att, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                next_c, next_h = self.cores[i * self.num_parallels + j](xt, prev_c, prev_h, prev_att_res)

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

        # batch_size * rnn_size
        top_h = outputs[-1]

        # x      : batch_size * input_size
        proj_w = F.sigmoid(self.proj_weight(x))

        logsoft = F.log_softmax(self.proj(top_h) * proj_w)

        return outputs, logsoft, proj_w


# with n num_layers
# new
class LSTM_DOUBLE_ATT_STACK_PARALLEL_A(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_A, self).__init__()

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
        for i in range(self.num_layers * 2):
            att = ATT.lstm_att_with_att_h(self.rnn_size, self.att_size)
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

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1)
            next_c = next_c.mean(1).view_as(prev_c)
            # next_c = next_c.max(1)[0].view_as(prev_c)

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1)
            next_h = next_h.mean(1).view_as(prev_h)
            # next_h = next_h.max(1)[0].view_as(prev_h)

            att_res = self.attens[i*2+1](att, next_h)

            # batch_size * rnn_size
            top_h = att_res + next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

        top_h = outputs[-1]
        logsoft = F.log_softmax(self.proj(top_h))

        return outputs, logsoft

# with n num_layers
# with multi proj
class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT, self).__init__()

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
        for i in range(self.num_layers * 2):
            att = ATT.lstm_att_with_x_att_h(self.rnn_size, self.att_size)
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

            prev_att_res = self.attens[i*2+0](xt, att, prev_h)

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

            att_res = self.attens[i*2+1](xt, att, next_h)

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

# with n num_layers
# with multi proj
class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_BU(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, bu_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_BU, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_parallels = num_parallels
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.bu_size = bu_size
        self.dropout = dropout

        # core
        self.cores = nn.ModuleList()
        for i in range(self.num_layers * self.num_parallels):
            core = CORE.lstm_core_with_att_bu(self.input_size, self.rnn_size)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers * 2):
            att = ATT.lstm_att_with_x_att_h(self.rnn_size, self.att_size)
            self.attens.append(att)

        # bu attention
        self.bu_attens = nn.ModuleList()
        for i in range(self.num_layers * 2):
            att = ATT.lstm_att_with_att_h(self.rnn_size, self.bu_size)
            self.bu_attens.append(att)

        # proj
        self.projs = nn.ModuleList()
        for i in range(self.num_layers):
            proj = nn.Linear(self.rnn_size, self.output_size)
            self.projs.append(proj)


    # x      : batch_size * input_size
    # att    : batch_size * att_size * rnn_size
    # prev_h : batch_size * rnn_size
    # prev_c : batch_size * rnn_size
    def forward(self, x, att, bu, inputs):

        outputs = []
        logprobs = []
        for i in range(self.num_layers):
            prev_c = inputs[i * 2]
            prev_h = inputs[i * 2 + 1]

            if i == 0:
                xt = x
            else:
                # xt = outputs[-1]
                xt = x + outputs[-1]

            prev_att_res = self.attens[i * 2 + 0](xt, att, prev_h)

            # batch_size * rnn_size
            prev_bu_res = self.bu_attens[i * 2 + 0](bu, prev_h)

            all_next_c = []
            all_next_h = []
            for j in range(self.num_parallels):
                next_c, next_h = self.cores[i * self.num_parallels + j](xt, prev_c, prev_h, prev_att_res, prev_bu_res)

                all_next_c.append(next_c)
                all_next_h.append(next_h)

            next_c = torch.cat([_.unsqueeze(1) for _ in all_next_c], 1)
            next_c = next_c.mean(1).view_as(prev_c)
            # next_c = next_c.max(1)[0].view_as(prev_c)

            next_h = torch.cat([_.unsqueeze(1) for _ in all_next_h], 1)
            next_h = next_h.mean(1).view_as(prev_h)
            # next_h = next_h.max(1)[0].view_as(prev_h)

            att_res = self.attens[i * 2 + 1](xt, att, next_h)

            bu_res = self.bu_attens[i * 2 + 1](bu, prev_h)

            # batch_size * rnn_size
            top_h = att_res + bu_res + next_h

            outputs.append(next_c)

            if self.dropout > 0:
                top_h = F.dropout(top_h, self.dropout)
            outputs.append(top_h)

            # policy
            logsoft = F.log_softmax(self.projs[i](top_h))

            logprobs.append(logsoft)

        return outputs, logprobs


# with n num_layers
# only one proj
class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_NEW(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_NEW, self).__init__()

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
        for i in range(self.num_layers * 2):
            att = ATT.lstm_att_with_x_att_h(self.rnn_size, self.att_size)
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

            prev_att_res = self.attens[i*2+0](xt, att, prev_h)

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

            att_res = self.attens[i*2+1](xt, att, next_h)

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


# with n num_layers
# lstm with mul layers
class LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_LSTM_MUL(nn.Module):

    def __init__(self, input_size, output_size, num_layers, num_parallels, rnn_size, att_size, dropout, block_num):
        super(LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_LSTM_MUL, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_parallels = num_parallels
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.dropout = dropout
        self.block_num = block_num

        # core
        self.cores = nn.ModuleList()
        for i in range(self.num_layers * self.num_parallels):
            core = CORE.lstm_mul_linear_core_with_att(self.input_size, self.rnn_size, self.num_layers, self.block_num)
            self.cores.append(core)

        # attention
        self.attens = nn.ModuleList()
        for i in range(self.num_layers * 2):
            att = ATT.lstm_att_with_x_att_h(self.rnn_size, self.att_size)
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

            prev_att_res = self.attens[i*2+0](xt, att, prev_h)

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

            att_res = self.attens[i*2+1](xt, att, next_h)

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