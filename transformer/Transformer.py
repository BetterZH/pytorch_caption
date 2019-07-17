import torch.nn as nn
import torch.nn.functional as F
import torch
import mixer.ReinforceSampler as ReinforceSampler
from torch.autograd import *
import torch.nn.init as init
import math
import numpy as np

def InitPositionEcoding(pos_size, model_size):

    # pos_enc: pos_size * model_size
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / model_size) for j in range(model_size)]
        if pos != 0 else np.zeros(model_size) for pos in range(pos_size)])

    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])

    # pos_size * model_size
    return torch.from_numpy(pos_enc).type(torch.FloatTensor)


# mask: batch_size * len_q * len_k
# seq_q: batch_size * len_q
# seq_k: batch_size * len_k
def getAttnPaddingMask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # batch_size * 1 * len_k
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # batch_size * len_q * len_k
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)

    # batch_size * len_q * len_k
    return pad_attn_mask


# just for self attention
# masks: batch_size * len_q
# seq_q: batch_size * len_q
def getSelfAttnPaddingMaskWithMask(seq_q, masks):

    batch_size, len_q = seq_q.size()

    # batch_size * 1 * len_q
    pad_attn_mask = masks.data.eq(0).unsqueeze(1)

    # batch_size * len_q * len_q
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_q)

    # batch_size * len_q * len_q
    return pad_attn_mask


# seq: batch_size * len_q
# output: batch_size * len_q * len_q
def getAttnSubsequentMask(seq):

    # seq: batch_size * len_q * len_q
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))

    # seq: batch_size * len_q * len_q
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # seq: batch_size * len_q * len_q
    subsequent_mask = torch.from_numpy(subsequent_mask)

    subsequent_mask = subsequent_mask.cuda()

    # output: batch_size * len_q * len_q
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, model_size, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = model_size ** 0.5

        self.dropout = nn.Dropout(dropout)

    # q : batch_size * len_q * k_size
    # k : batch_size * len_k * k_size
    # v : batch_size * len_v * v_size
    # output: batch_size * len_q * v_size
    # attn_mask: batch_size * len_q
    def forward(self, q, k, v, attn_mask=None):

        # MatMul
        # batch_size * len_q * len_k
        attn = torch.bmm(q, k.transpose(1, 2))

        # Scale
        # batch_size * len_q * len_k
        attn = attn / self.temperature

        # Mask (opt.)
        # attn_mask: batch_size * len_q * len_k
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))

        # SoftMax
        # batch_size * len_q * len_k
        attn = F.softmax(attn, -1)

        # Dropout
        attn = self.dropout(attn)

        # att: batch_size * len_q * len_k
        # v: batch_size * len_v * v_size
        # assert len_k == len_v
        # output: batch_size * len_q * v_size
        output = torch.bmm(attn, v)

        # output: batch_size * len_q * v_size
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, model_size, k_size, v_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.model_size = model_size
        self.k_size = k_size
        self.v_size = v_size

        self.attention = ScaledDotProductAttention(self.model_size)

        self.linear_q = nn.Linear(self.model_size, k_size)
        self.linear_k = nn.Linear(self.model_size, k_size)
        self.linear_v = nn.Linear(self.model_size, v_size)

        self.proj = nn.Linear(self.head_size*self.v_size, self.model_size)

        self.dropout = nn.Dropout(dropout)


    # q : batch_size * len_q * k_size
    # k : batch_size * len_k * k_size
    # v : batch_size * len_v * v_size
    # output: batch_size * len_q * model_size
    # attn_mask: batch_size * len_q * len_k
    def forward(self, q, k, v, attn_mask=None):

        batch_size, len_q, model_size = q.size()
        batch_size, len_k, model_size = k.size()
        batch_size, len_v, model_size = v.size()

        # (head_size*batch_size) * len_q * model_size
        q = q.repeat(self.head_size, 1, 1)
        # head_size * (batch_size*len_q) * model_size
        q = q.view(self.head_size, -1, self.model_size)

        # (head_size*batch_size) * len_k * model_size
        k = k.repeat(self.head_size, 1, 1)
        # head_size * (batch_size*len_k) * model_size
        k = k.view(self.head_size, -1, self.model_size)

        # (head_size*batch_size) * len_v * model_size
        v = v.repeat(self.head_size, 1, 1)
        # head_size * (batch_size*len_v) * model_size
        v = v.view(self.head_size, -1, self.model_size)

        # head_size * (batch_size*len_q) * k_size
        q = self.linear_q(q)
        # head_size * (batch_size*len_k) * k_size
        k = self.linear_k(k)
        # head_size * (batch_size*len_v) * v_size
        v = self.linear_v(v)

        # (head_size*batch_size) * len_q * k_size
        q = q.view(-1, len_q, self.k_size)
        # (head_size*batch_size) * len_k * k_size
        k = k.view(-1, len_k, self.k_size)
        # (head_size*batch_size) * len_v * v_size
        v = v.view(-1, len_v, self.v_size)

        # (head_size*batch_size) * len_q * v_size
        # attn_mask: batch_size * len_q * len_k
        # attn_mask_repeat: (head_size*batch_size) * len_q * len_k
        attn_mask_repeat = None
        if attn_mask is not None:
            attn_mask_repeat = attn_mask.repeat(self.head_size, 1, 1)

        # q: (head_size*batch_size) * len_q * k_size
        # k: (head_size*batch_size) * len_k * k_size
        # v: (head_size*batch_size) * len_v * v_size
        # output: (head_size*batch_size) * len_q * v_size
        output = self.attention(q, k, v, attn_mask=attn_mask_repeat)

        # batch_size * len_q * (head_size*v_size)
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)

        # batch_size * len_q * model_size
        output = self.proj(output)

        # batch_size * len_q * model_size
        output = self.dropout(output)

        # batch_size * len_q * model_size
        return output


class PositionwiseFeedForward1(nn.Module):
    def __init__(self, model_size, inner_layer_size, dropout=0.1):
        super(PositionwiseFeedForward1, self).__init__()
        self.layer1 = nn.Linear(model_size, inner_layer_size)
        self.layer2 = nn.Linear(inner_layer_size, model_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    # input: batch_size * len_q * model_size
    # output: batch_size * len_q * model_size
    def forward(self, input):

        # input: batch_size * len_q * model_size
        # output: batch_size * len_q * inner_layer_size
        output = self.layer1(input)

        # batch_size * len_q * inner_layer_size
        output = self.relu(output)

        # batch_size * len_q * model_size
        output = self.layer2(output)

        # dropout
        output = self.dropout(output)

        # batch_size * len_q * model_size
        return output


class PositionwiseFeedForward(nn.Module):

    def __init__(self, model_size, inner_layer_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer1 = nn.Conv1d(model_size, inner_layer_size, 1)
        self.layer2 = nn.Conv1d(inner_layer_size, model_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # input: batch_size * len_q * model_size
    # output: batch_size * len_q * model_size
    def forward(self, input):

        # <- batch_size * len_q * model_size
        # -> batch_size * model_size * len_q
        input = input.transpose(1, 2)

        # <- batch_size * model_size * len_q
        # -> batch_size * inner_layer_size * len_q
        output = self.layer1(input)

        # batch_size * inner_layer_size * model_size
        output = self.relu(output)

        # <- batch_size * inner_layer_size * len_q
        # -> batch_size * model_size * len_q
        output = self.layer2(output)

        # <- batch_size * model_size * len_q
        # -> batch_size * len_q * model_size
        output = output.transpose(1, 2)

        # dropout
        output = self.dropout(output)

        # batch_size * len_q * model_size
        return output


class LayerNormalization(nn.Module):

    def __init__(self, model_size, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(model_size), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(model_size), requires_grad=True)

    # batch_size * len_q * model_size
    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class BatchNorm1d(nn.Module):

    def __init__(self, model_size):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(model_size)

    # batch_size * len_q * model_size
    def forward(self, x):
        # batch_size * model_size * len_q
        x = x.transpose(1, 2).contiguous()
        # batch_size * model_size * len_q
        x = self.bn(x)
        # batch_size * len_q * model_size
        x = x.transpose(1, 2)
        return x


# class Encoder(nn.Module):
#     def __init__(self, vocab_size, seq_len, head_size=8, model_size=512,
#                  n_layers=6, k_size=64, v_size=64, inner_layer_size=2048):
#         super(Encoder, self).__init__()
#
#         pos_size = seq_len + 1
#
#         self.pos_embed = nn.Embedding(pos_size, model_size)
#         self.pos_embed.weight.data = InitPositionEcoding(pos_size, model_size)
#
#         self.word_embed = nn.Embedding(vocab_size, model_size)
#
#         # input:  batch_size * len_q * model_size
#         # output: batch_size * len_q * model_size
#         # attn_mask: batch_size * len_q * len_q
#         self.encoder_layers = nn.ModuleList([
#             EncoderLayer(head_size, model_size, k_size, v_size, inner_layer_size)
#             for _ in range(n_layers)
#         ])
#
#     # input_seq: batch_size * len_q
#     # input_pos: batch_size * len_q
#     def forward(self, input_seq, input_pos):
#
#         # input_seq: batch_size * len_q
#         # input_enc: batch_size * len_q
#         input_enc = self.word_embed(input_seq)
#
#         # input_pos: batch_size * len_q * model_size
#         # input_enc: batch_size * len_q * model_size
#         input_enc += self.pos_embed(input_pos)
#
#         # input_seq: batch_size * len_q
#         # attn_mask: batch_size * len_q * len_q
#         attn_mask = getAttnPaddingMask(input_seq, input_seq)
#
#         output_enc = input_enc
#         for enc_layer in self.encoder_layers:
#             # input:  batch_size * len_q * model_size
#             # output: batch_size * len_q * model_size
#             # attn_mask: batch_size * len_q * len_k
#             output_enc = enc_layer(output_enc, attn_mask)
#
#         # output_enc: batch_size * len_q * model_size
#         return output_enc

# class Decoder(nn.Module):
#     def __init__(self, vocab_size, seq_len, head_size=8, model_size=512,
#                  n_layers=6, k_size=64, v_size=64, inner_layer_size=2048):
#         super(Decoder, self).__init__()
#
#         pos_size = seq_len + 1
#
#         self.pos_embed = nn.Embedding(pos_size, model_size)
#         self.pos_embed.weight.data = InitPositionEcoding(pos_size, model_size)
#
#         # batch_size * vocab_size * model_sizes
#         self.word_embed = nn.Embedding(vocab_size, model_size)
#
#         # input_dec:  batch_size * len_q * model_size
#         # output_enc: batch_size * len_q1 * model_size
#         # output:     batch_size * len_q * model_size
#         self.dec_layers = nn.ModuleList([
#             DecoderLayer(head_size, model_size, k_size, v_size, inner_layer_size)
#             for _ in range(n_layers)
#         ])
#
#     # target_seq: batch_size * len_q
#     # target_pos: batch_size * len_q
#     # output_enc: batch_size * len_q * model_size
#     def forward(self, target_seq, target_pos, input_seq, output_enc):
#
#         # input_dec: batch_size * len_q * model_size
#         input_dec = self.word_embed(target_seq)
#
#         # input_dec: batch_size * len_q * model_size
#         input_dec += self.pos_embed(target_pos)
#
#         # Attention
#         # target_seq: batch_size * len_q
#         # dec_self_attn_pad_mask: batch_size * len_q * len_q
#         dec_self_attn_pad_mask = getAttnPaddingMask(target_seq, target_seq)
#
#         # dec_self_attn_sub_mask: batch_size * len_q * len_q
#         dec_self_attn_sub_mask = getAttnSubsequentMask(target_seq)
#
#         # dec_self_attn_mask: batch_size * len_q * len_q
#         dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_sub_mask, 0)
#
#         # dec_enc_attn_pad_mask: batch_size * len_q * len_q
#         dec_enc_attn_pad_mask = getAttnPaddingMask(target_seq, input_seq)
#
#         # output_dec: batch_size * len_q * model_size
#         output_dec = input_dec
#         for dec_layer in self.dec_layers:
#             # output_dec: batch_size * len_q * model_size
#             # output_enc: batch_size * len_q * model_size
#             # output_dec: batch_size * len_q * model_size
#             # dec_self_attn_mask: batch_size * len_q * len_q
#             # dec_enc_attn_pad_mask: batch_size * len_q * len_q
#             output_dec = dec_layer(output_dec, output_enc,
#                                    slf_attn_mask = dec_self_attn_mask,
#                                    dec_enc_attn_mask = dec_enc_attn_pad_mask)
#
#         # output_dec: batch_size * len_q * model_size
#         return output_dec
#
#
# class Transformer(nn.Module):
#
#     def __init__(self, model_size, vocab_size, seq_len, dropout):
#         super(Transformer, self).__init__()
#
#         self.model_size = model_size
#         self.vocab_size = vocab_size
#         self.seq_len = seq_len
#         self.dropout = dropout
#
#         # <- input_seq: batch_size * len_q
#         # <- input_pos: batch_size * len_q
#         # -> batch_size * len_q * model_size
#         self.encoder = Encoder(vocab_size, seq_len)
#
#         # <- batch_size * len_q
#         # <- batch_size * len_q
#         # -> batch_size * len_q * model_size
#         self.decoder = Decoder(vocab_size, seq_len)
#
#         # <- batch_size * len_q * model_size
#         # -> batch_size * len_q * vocab_size
#         self.proj = nn.Linear(self.model_size, self.vocab_size)
#
#         # share weight
#         # word_embed: batch_size * vocab_size * model_size
#         self.proj.linear.weight = self.decoder.word_embed.weight
#         self.encoder.word_embed.weight = self.decoder.word_embed.weight
#
#
#     def get_trainable_parameters(self):
#         enc_freezed_param_ids = set(map(id, self.encoder.pos_embed.parameters()))
#         dec_freezed_param_ids = set(map(id, self.decoder.pos_embed.parameters()))
#         freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
#         return (p for p in self.parameters() if id(p) not in freezed_param_ids)
#
#     # input_seq:  batch_size * len_q
#     # target_seq: batch_size * len_q
#     def forward(self, input_seq, target_seq):
#
#         # input_pos:  batch_size * len_q
#         input_pos = np.array([
#             [pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(seq)]
#             for seq in input_seq])
#         input_pos = torch.LongTensor(input_pos).cuda()
#         input_pos = Variable(input_pos, requires_grad=False)
#
#         # target_pos: batch_size * len_q
#         target_pos = np.array([
#             [pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(seq)]
#             for seq in target_seq])
#         target_pos = torch.LongTensor(target_pos).cuda()
#         target_pos = Variable(target_pos, requires_grad=False)
#
#         # input_seq: batch_size * len_q
#         # input_pos: batch_size * len_q
#         # output_enc: batch_size * len_q * model_size
#         output_enc = self.encoder(input_seq, input_pos)
#
#         # target_seq: batch_size * len_q
#         # target_pos: batch_size * len_q
#         # input_seq:  batch_size * len_q
#         # output_enc: batch_size * len_q * model_size
#         # output_dec: batch_size * len_q * model_size
#         output_dec = self.decoder(target_seq, target_pos, input_seq, output_enc)
#
#         if self.dropout > 0:
#             output_dec = F.dropout(output_dec, self.dropout)
#
#         # output_dec:   batch_size * len_q * model_size
#         # seq_logsofts: batch_size * len_q * output_size
#         seq_logsofts = F.log_softmax(self.proj(output_dec), -1)
#
#         # seq_logsofts: batch_size * len_q * output_size
#         return seq_logsofts









