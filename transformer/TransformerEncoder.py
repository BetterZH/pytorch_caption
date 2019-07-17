from transformer.Transformer import *

class EncoderLayer(nn.Module):
    def __init__(self, head_size, model_size, k_size, v_size, inner_layer_size):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(head_size, model_size, k_size, v_size)
        self.feedForward = PositionwiseFeedForward(model_size, inner_layer_size)

        self.norm1 = LayerNormalization(model_size)
        self.norm2 = LayerNormalization(model_size)


    # x:  batch_size * len_q * model_size
    # output: batch_size * len_q * model_size
    # attn_mask: batch_size * len_q * len_k
    def forward(self, x, attn_mask=None):

        # q : batch_size * len_q * k_size
        # k : batch_size * len_k * k_size
        # v : batch_size * len_v * v_size
        # x1: batch_size * len_q * model_size
        x1 = self.slf_attn(x, x, x, attn_mask=attn_mask) + x

        # x1: batch_size * len_q * model_size
        x1 = self.norm1(x1)

        # x1: batch_size * len_q * model_size
        output = self.feedForward(x1) + x1

        # x1: batch_size * len_q * model_size
        output = self.norm2(output)

        # batch_size * len_q * model_size
        return output


class ImageEncoder(nn.Module):
    def __init__(self, att_feat_size, head_size=8, model_size=512,
                 n_layers=6, k_size=64, v_size=64, inner_layer_size=2048):
        super(ImageEncoder, self).__init__()

        # pos_size = seq_len + 1
        #
        # self.pos_embed = nn.Embedding(pos_size, model_size)
        # self.pos_embed.weight.data = InitPositionEcoding(pos_size, model_size)

        self.image_embed = nn.Linear(att_feat_size, model_size)

        # input:  batch_size * len_q * model_size
        # output: batch_size * len_q * model_size
        # attn_mask: batch_size * len_q * len_q
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(head_size, model_size, k_size, v_size, inner_layer_size)
            for _ in range(n_layers)
        ])

    # att_feats: batch_size * len_q * att_feat_size
    # input_pos: batch_size * len_q
    def forward(self, att_feats):

        # att_feats: batch_size * len_q * att_feat_size
        # input_enc: batch_size * len_q * model_size
        input_enc = self.image_embed(att_feats)

        # input_pos: batch_size * len_q * model_size
        # input_enc: batch_size * len_q * model_size
        # input_enc += self.pos_embed(input_pos)

        output_enc = input_enc
        for enc_layer in self.encoder_layers:
            # input:  batch_size * len_q * model_size
            # output: batch_size * len_q * model_size
            # attn_mask: batch_size * len_q * len_k
            output_enc = enc_layer(output_enc)

        # output_enc: batch_size * len_q * model_size
        return output_enc