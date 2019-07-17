from transformer.Transformer import *
from transformer.TransformerAttention import *
import transformer.transformer_utils as transformer_utils

class ImageMoreSupDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageMoreSupDecoder, self).__init__()

        self.vocab_size = opt.vocab_size
        # words (with out BOS EOS)
        self.seq_length = opt.seq_length
        self.head_size = opt.head_size
        self.model_size = opt.model_size
        self.n_layers = opt.n_layers
        self.k_size = opt.k_size
        self.v_size = opt.v_size
        self.inner_layer_size = opt.inner_layer_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.n_layers_output = opt.n_layers_output


        self.pos_size = self.seq_length + 2

        self.pos_embed = nn.Embedding(self.pos_size, self.model_size)
        self.pos_embed.weight.data = InitPositionEcoding(self.pos_size, self.model_size)

        # batch_size * vocab_size * model_sizes
        self.word_embed = nn.Embedding(self.vocab_size + 1, self.model_size)

        # input_dec:  batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # output:     batch_size * len_q * model_size
        print('transformer_decoder_layer_type: ' + opt.transformer_decoder_layer_type)
        self.dec_layers = nn.ModuleList([
            transformer_utils.get_transformer_decoder_layer(opt)
            for _ in range(self.n_layers)
        ])

        self.projs = nn.ModuleList([
            nn.Linear(self.model_size, self.vocab_size + 1)
            for _ in range(self.n_layers_output)
        ])


    # target_seq: batch_size * len_q
    # target_pos: batch_size * len_q
    # output_enc: batch_size * len_q * model_size
    # masks: batch_size * len_q
    def forward(self, target_seq, target_pos, output_enc, masks):

        # input_dec: batch_size * len_q * model_size
        new_masks = masks.unsqueeze(2).repeat(1, 1, self.model_size)

        # input_dec: batch_size * len_q * model_size
        input_dec = self.word_embed(target_seq)

        # input_dec: batch_size * len_q * model_size
        input_dec += self.pos_embed(target_pos)

        # input_dec: batch_size * len_q * model_size
        input_dec = input_dec * new_masks

        # Attention
        # target_seq: batch_size * len_q
        # dec_self_attn_pad_mask: batch_size * len_q * len_q
        dec_self_attn_pad_mask = getSelfAttnPaddingMaskWithMask(target_seq, masks)

        # dec_self_attn_sub_mask: batch_size * len_q * len_q
        dec_self_attn_sub_mask = getAttnSubsequentMask(target_seq)

        # dec_self_attn_mask: batch_size * len_q * len_q
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_sub_mask, 0)

        # output_dec: batch_size * len_q * model_size
        output_dec = input_dec

        logprobs = []

        for i in range(self.n_layers):
            # output_dec: batch_size * len_q * model_size
            # output_enc: batch_size * len_q * model_size
            # output_dec: batch_size * len_q * model_size
            # dec_self_attn_mask: batch_size * len_q * len_q
            # dec_enc_attn_pad_mask: batch_size * len_q * len_q
            output_dec = self.dec_layers[i](output_dec, output_enc,
                                            slf_attn_mask=dec_self_attn_mask,
                                            dec_enc_attn_mask=None)

            if self.drop_prob_lm > 0:
                output_dec = F.dropout(output_dec, self.drop_prob_lm)

            j = i - (self.n_layers - self.n_layers_output)
            if j >= 0 and j <= self.n_layers_output:
                logsoft = F.log_softmax(self.projs[j](output_dec), -1)
                logprobs.append(logsoft)

            # output_dec: batch_size * len_q * model_size
            # output_dec = input_dec + output_dec

        # output_dec: n_layers * batch_size * len_q * model_size
        return logprobs



class ImageCMSDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageCMSDecoder, self).__init__()

        self.vocab_size = opt.vocab_size
        # words (with out BOS EOS)
        self.seq_length = opt.seq_length
        self.head_size = opt.head_size
        self.model_size = opt.model_size
        self.n_layers = opt.n_layers
        self.k_size = opt.k_size
        self.v_size = opt.v_size
        self.inner_layer_size = opt.inner_layer_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.n_layers_output = opt.n_layers_output


        self.pos_size = self.seq_length + 2 + 1

        self.pos_embed = nn.Embedding(self.pos_size, self.model_size)
        self.pos_embed.weight.data = InitPositionEcoding(self.pos_size, self.model_size)

        # batch_size * vocab_size * model_sizes
        self.word_embed = nn.Embedding(self.vocab_size + 1, self.model_size)

        # input_dec:  batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # output:     batch_size * len_q * model_size
        print('transformer_decoder_layer_type: ' + opt.transformer_decoder_layer_type)
        self.dec_layers = nn.ModuleList([
            transformer_utils.get_transformer_decoder_layer(opt)
            for _ in range(self.n_layers)
        ])

        self.projs = nn.ModuleList([
            nn.Linear(self.model_size, self.vocab_size + 1)
            for _ in range(self.n_layers_output)
        ])


    # target_seq: batch_size * len_q
    # target_pos: batch_size * len_q
    # fc_feats: batch_size * model_size
    # att_feats: batch_size * len_q * model_size
    # masks: batch_size * len_q
    def forward(self, target_seq, target_pos, fc_feats, att_feats, masks):

        # input_dec: batch_size * len_q * model_size
        new_masks = masks.unsqueeze(2).repeat(1, 1, self.model_size)

        # input_dec: batch_size * len_q * model_size
        input_dec = self.word_embed(target_seq)

        # combine image feat
        # <- batch_size * 1 * model_size
        # <- batch_size * len_q * model_size
        # -> batch_size * len_1_q * model_size
        input_dec = torch.cat([fc_feats.unsqueeze(1), input_dec], 1)

        # input_dec: batch_size * len_1_q * model_size
        input_dec += self.pos_embed(target_pos)

        # input_dec: batch_size * len_1_q * model_size
        input_dec = input_dec * new_masks

        # Attention
        # target_seq: batch_size * len_q
        # dec_self_attn_pad_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_pad_mask = getSelfAttnPaddingMaskWithMask(target_pos, masks)

        # dec_self_attn_sub_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_sub_mask = getAttnSubsequentMask(target_pos)

        # dec_self_attn_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_sub_mask, 0)

        # output_dec: batch_size * len_1_q * model_size
        output_dec = input_dec

        logprobs = []

        for i in range(self.n_layers):
            # output_dec: batch_size * len_1_q * model_size
            # output_enc: batch_size * len_1_q * model_size
            # output_dec: batch_size * len_1_q * model_size
            # dec_self_attn_mask: batch_size * len_1_q * len_1_q
            # dec_enc_attn_pad_mask: batch_size * len_1_q * len_1_q
            output_dec = self.dec_layers[i](output_dec, att_feats,
                                            slf_attn_mask=dec_self_attn_mask,
                                            dec_enc_attn_mask=None)

            if self.drop_prob_lm > 0:
                output_dec = F.dropout(output_dec, self.drop_prob_lm)

            j = i - (self.n_layers - self.n_layers_output)
            if j >= 0 and j <= self.n_layers_output:
                logsoft = F.log_softmax(self.projs[j](output_dec), -1)
                # logsoft: batch_size * len_q * model_size
                logsoft = logsoft[:, 1:, :]
                logsoft = logsoft.contiguous()
                logprobs.append(logsoft)


            # output_dec: batch_size * len_q * model_size
            # output_dec = input_dec + output_dec

        # output_dec: n_layers * batch_size * len_q * model_size
        return logprobs


class ImageCRMSDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageCRMSDecoder, self).__init__()

        self.vocab_size = opt.vocab_size
        # words (with out BOS EOS)
        self.seq_length = opt.seq_length
        self.head_size = opt.head_size
        self.model_size = opt.model_size
        self.n_layers = opt.n_layers
        self.k_size = opt.k_size
        self.v_size = opt.v_size
        self.inner_layer_size = opt.inner_layer_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.n_layers_output = opt.n_layers_output

        self.pos_size = self.seq_length + 2 + 1

        self.pos_embed = nn.Embedding(self.pos_size, self.model_size)
        self.pos_embed.weight.data = InitPositionEcoding(self.pos_size, self.model_size)

        # batch_size * vocab_size * model_sizes
        self.word_embed = nn.Embedding(self.vocab_size + 1, self.model_size)

        # input_dec:  batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # output:     batch_size * len_q * model_size
        print('transformer_decoder_layer_type: ' + opt.transformer_decoder_layer_type)
        self.dec_layers = nn.ModuleList([
            transformer_utils.get_transformer_decoder_layer(opt)
            for _ in range(self.n_layers)
        ])

        self.projs = nn.ModuleList([
            nn.Linear(self.model_size, self.vocab_size + 1)
            for _ in range(self.n_layers_output)
        ])

    # target_seq: batch_size * len_q
    # target_pos: batch_size * len_q
    # fc_feats: batch_size * model_size
    # att_feats: batch_size * len_q * model_size
    # masks: batch_size * len_q
    def forward(self, target_seq, target_pos, fc_feats, att_feats, masks):

        # input_dec: batch_size * len_q * model_size
        new_masks = masks.unsqueeze(2).repeat(1, 1, self.model_size)

        # input_dec: batch_size * len_q * model_size
        input_dec = self.word_embed(target_seq)

        # combine image feat
        # <- batch_size * 1 * model_size
        # <- batch_size * len_q * model_size
        # -> batch_size * len_1_q * model_size
        input_dec = torch.cat([fc_feats.unsqueeze(1), input_dec], 1)

        # input_dec: batch_size * len_1_q * model_size
        input_dec += self.pos_embed(target_pos)

        # input_dec: batch_size * len_1_q * model_size
        input_dec = input_dec * new_masks

        # Attention
        # target_seq: batch_size * len_q
        # dec_self_attn_pad_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_pad_mask = getSelfAttnPaddingMaskWithMask(target_pos, masks)

        # dec_self_attn_sub_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_sub_mask = getAttnSubsequentMask(target_pos)

        # dec_self_attn_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_sub_mask, 0)

        # output_dec: batch_size * len_1_q * model_size
        output_dec = input_dec

        logprobs = []

        for i in range(self.n_layers):

            if i > 0:
                output_dec = output_dec + input_dec

            # output_dec: batch_size * len_1_q * model_size
            # output_enc: batch_size * len_1_q * model_size
            # output_dec: batch_size * len_1_q * model_size
            # dec_self_attn_mask: batch_size * len_1_q * len_1_q
            # dec_enc_attn_pad_mask: batch_size * len_1_q * len_1_q
            output_dec = self.dec_layers[i](output_dec, att_feats,
                                            slf_attn_mask=dec_self_attn_mask,
                                            dec_enc_attn_mask=None)

            if self.drop_prob_lm > 0:
                output_dec = F.dropout(output_dec, self.drop_prob_lm)

            j = i - (self.n_layers - self.n_layers_output)
            if j >= 0 and j <= self.n_layers_output:
                logsoft = F.log_softmax(self.projs[j](output_dec), -1)
                # logsoft: batch_size * len_q * model_size
                logsoft = logsoft[:, 1:, :]
                logsoft = logsoft.contiguous()
                logprobs.append(logsoft)

                # output_dec: batch_size * len_q * model_size
                # output_dec = input_dec + output_dec

        # output_dec: n_layers * batch_size * len_q * model_size
        return logprobs

class ImageMoreSupAttentionDecoder(ImageMoreSupDecoder):
    def __init__(self, opt):
        super(ImageMoreSupAttentionDecoder, self).__init__(opt)