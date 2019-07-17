from transformer.Transformer import *
from transformer.Attention import *

class DecoderLayer(nn.Module):
    def __init__(self, opt):
        super(DecoderLayer, self).__init__()

        head_size = opt.head_size
        model_size = opt.model_size
        k_size = opt.k_size
        v_size = opt.v_size
        inner_layer_size = opt.inner_layer_size
        drop_prob_lm = opt.drop_prob_lm

        # head_size, model_size, k_size, v_size
        self.slf_attn = MultiHeadAttention(head_size, model_size, k_size, v_size, drop_prob_lm)
        self.enc_attn = MultiHeadAttention(head_size, model_size, k_size, v_size, drop_prob_lm)
        self.feedForwad = PositionwiseFeedForward(model_size, inner_layer_size, drop_prob_lm)

        if opt.norm_type == 0:
            norm = LayerNormalization
        elif opt.norm_type == 1:
            norm = BatchNorm1d

        self.norm1 = norm(model_size)
        self.norm2 = norm(model_size)
        self.norm3 = norm(model_size)

    # input_dec:  batch_size * len_q * model_size
    # output_enc: batch_size * len_q1 * model_size
    # output:     batch_size * len_q * model_size
    def forward(self, input_dec, output_enc, slf_attn_mask=None, dec_enc_attn_mask=None):

        # batch_size * len_q * model_size
        x1 = self.slf_attn(input_dec, input_dec, input_dec, attn_mask=slf_attn_mask) + input_dec

        # batch_size * len_q * model_size
        x1 = self.norm1(x1)

        # x1: batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # x2: batch_size * len_q * model_size
        # q k v
        x2 = self.enc_attn(x1, output_enc, output_enc, attn_mask=dec_enc_attn_mask) + x1

        # batch_size * len_q * model_size
        x2 = self.norm2(x2)

        # batch_size * len_q * model_size
        x3 = self.feedForwad(x2) + x2

        # batch_size * len_q * model_size
        x3 = self.norm3(x3)

        # batch_size * len_q * model_size
        return x3


class CDecoderLayer(nn.Module):
    def __init__(self, opt):
        super(CDecoderLayer, self).__init__()

        head_size = opt.head_size
        model_size = opt.model_size
        k_size = opt.k_size
        v_size = opt.v_size
        inner_layer_size = opt.inner_layer_size
        drop_prob_lm = opt.drop_prob_lm

        # head_size, model_size, k_size, v_size
        self.slf_attn = MultiHeadAttention(head_size, model_size, k_size, v_size, drop_prob_lm)
        self.enc_attn = MultiHeadAttention(head_size, model_size, k_size, v_size, drop_prob_lm)
        self.feedForwad = PositionwiseFeedForward(model_size, inner_layer_size, drop_prob_lm)

        if opt.norm_type == 0:
            norm = LayerNormalization
        elif opt.norm_type == 1:
            norm = BatchNorm1d

        self.norm1 = norm(model_size)
        self.norm2 = norm(model_size)
        self.norm3 = norm(model_size)

    # input_dec:  batch_size * len_1_q * model_size
    # output_enc: batch_size * len_q1 * model_size
    # output:     batch_size * len_q * model_size
    def forward(self, input_dec, output_enc, slf_attn_mask=None, dec_enc_attn_mask=None):

        # batch_size * len_1_q * model_size
        # batch_size * len_1_q * len_1_q
        # slf_attn_mask: batch_size * len_1_q * len_1_q
        # <- q, k, v
        x1 = self.slf_attn(input_dec, input_dec, input_dec, attn_mask=slf_attn_mask) + input_dec

        # batch_size * len_1_q * model_size
        x1 = self.norm1(x1)

        # x1: batch_size * len_1_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # x2: batch_size * len_1_q * model_size
        # q k v
        x2 = self.enc_attn(x1, output_enc, output_enc, attn_mask=dec_enc_attn_mask) + x1

        # batch_size * len_1_q * model_size
        x2 = self.norm2(x2)

        # batch_size * len_1_q * model_size
        x3 = self.feedForwad(x2) + x2

        # batch_size * len_1_q * model_size
        x3 = self.norm3(x3)

        return x3


class AttentionDecoderLayer(nn.Module):
    def __init__(self, opt):
        super(AttentionDecoderLayer, self).__init__()

        head_size = opt.head_size
        model_size = opt.model_size
        k_size = opt.k_size
        v_size = opt.v_size
        inner_layer_size = opt.inner_layer_size
        drop_prob_lm = opt.drop_prob_lm

        # head_size, model_size, k_size, v_size
        self.slf_attn = MultiHeadAttention(head_size, model_size, k_size, v_size, drop_prob_lm)
        self.enc_attn = MultiHeadAttention(head_size, model_size, k_size, v_size, drop_prob_lm)
        self.feedForwad = PositionwiseFeedForward(model_size, inner_layer_size, drop_prob_lm)
        self.soft_attn = SoftAttention(model_size)

        if opt.norm_type == 0:
            norm = LayerNormalization
        elif opt.norm_type == 1:
            norm = BatchNorm1d

        self.norm1 = norm(model_size)
        self.norm2 = norm(model_size)
        self.norm3 = norm(model_size)
        self.soft_norm = norm(model_size)


    # input_dec:  batch_size * len_q * model_size
    # output_enc: batch_size * len_q1 * model_size
    # output:     batch_size * len_q * model_size
    def forward(self, input_dec, output_enc, slf_attn_mask=None, dec_enc_attn_mask=None):

        # batch_size * len_q * model_size
        x1 = self.slf_attn(input_dec, input_dec, input_dec, attn_mask=slf_attn_mask) + input_dec

        # batch_size * len_q * model_size
        x1 = self.norm1(x1)

        # soft attention
        # batch_size * len_q * model_size
        x1 = self.soft_attn(output_enc, x1) + x1

        x1 = self.soft_norm(x1)

        # x1: batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # x2: batch_size * len_q * model_size
        x2 = self.enc_attn(x1, output_enc, output_enc, attn_mask=dec_enc_attn_mask) + x1

        # batch_size * len_q * model_size
        x2 = self.norm2(x2)

        # batch_size * len_q * model_size
        x3 = self.feedForwad(x2) + x2

        # batch_size * len_q * model_size
        x3 = self.norm3(x3)

        # batch_size * len_q * model_size
        return x3
