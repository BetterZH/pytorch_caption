import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftAttention(nn.Module):

    def __init__(self, model_size):
        super(SoftAttention, self).__init__()

        self.model_size = model_size

        self.a2a = nn.Linear(self.model_size, self.model_size)
        self.h2a = nn.Linear(self.model_size, self.model_size)
        self.d2d = nn.Linear(self.model_size, 1)


    # att : batch * att_size * model_size
    # h   : batch * h_size * model_size
    def forward(self, att, h):

        batch_size = att.size(0)
        att_size = att.size(1)
        model_size = att.size(2)

        h_size = h.size(1)

        # ##################################################
        # spatial attention start

        # att_v_1
        # batch * att_size * model_size
        att_v_1 = self.a2a(att)
        # batch * att_size * 1 * model_size
        att_v_1 = att_v_1.unsqueeze(2)
        # batch * att_size * h_size * model_size
        att_v_1 = att_v_1.expand(batch_size, att_size, h_size, model_size)

        # att_h_1
        # batch * h_size * model_size
        att_h = self.h2a(h)
        # batch * 1 * h_size * model_size
        att_h_1 = att_h.unsqueeze(1)
        # batch * att_size * h_size * model_size
        att_h_1 = att_h_1.expand_as(att_v_1)

        # batch * att_size * h_size * model_size
        dot = att_v_1 + att_h_1
        # batch * att_size * h_size * model_size
        dot = F.tanh(dot)
        # batch * att_size * h_size * 1
        dot = self.d2d(dot)
        # batch * att_size * h_size
        dot = dot.view(batch_size, att_size, h_size)

        # batch_size * att_size * h_size
        weight = F.softmax(dot, 1)

        # batch_size * model_size * att_size
        att_t = att.transpose(1, 2)

        # att_t: batch_size * model_size * att_size
        # weight: batch_size * att_size * h_size
        # att_res: batch_size * model_size * h_size
        att_res = torch.bmm(att_t, weight)
        # spatial attention end
        # ##################################################

        # batch_size * h_size * model_size
        att_res = att_res.transpose(1, 2)

        # batch_size * h_size * model_size
        return att_res