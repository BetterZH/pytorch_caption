from transformer.Transformer import *
from transformer.transformer_utils import *

class ImageDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageDecoder, self).__init__()

        vocab_size = opt.vocab_size
        # words (with out EOS EOS)
        seq_length = opt.seq_length
        model_size = opt.model_size
        n_layers = opt.n_layers

        pos_size = seq_length + 2

        self.model_size = model_size

        self.pos_embed = nn.Embedding(pos_size, model_size)
        self.pos_embed.weight.data = InitPositionEcoding(pos_size, model_size)

        # batch_size * vocab_size * model_sizes
        self.word_embed = nn.Embedding(vocab_size + 1, model_size)

        # input_dec:  batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # output:     batch_size * len_q * model_size
        print('transformer_decoder_layer_type: ' + opt.transformer_decoder_layer_type)
        self.dec_layers = nn.ModuleList([
            get_transformer_decoder_layer(opt)
            for _ in range(n_layers)
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
        for dec_layer in self.dec_layers:
            # output_dec: batch_size * len_q * model_size
            # output_enc: batch_size * len_q * model_size
            # output_dec: batch_size * len_q * model_size
            # dec_self_attn_mask: batch_size * len_q * len_q
            # dec_enc_attn_pad_mask: batch_size * len_q * len_q
            output_dec = dec_layer(output_dec, output_enc,
                                   slf_attn_mask = dec_self_attn_mask,
                                   dec_enc_attn_mask = None)

        # output_dec: batch_size * len_q * model_size
        return output_dec


class ImageCDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageCDecoder, self).__init__()

        vocab_size = opt.vocab_size
        # words (with out EOS EOS)
        seq_length = opt.seq_length
        model_size = opt.model_size
        n_layers = opt.n_layers

        # for image BOS
        # pos start from 1
        pos_size = seq_length + 2 + 1

        self.model_size = model_size

        self.pos_embed = nn.Embedding(pos_size, model_size)
        self.pos_embed.weight.data = InitPositionEcoding(pos_size, model_size)

        # batch_size * vocab_size * model_sizes
        self.word_embed = nn.Embedding(vocab_size + 1, model_size)

        # input_dec:  batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # output:     batch_size * len_q * model_size
        print('transformer_decoder_layer_type: ' + opt.transformer_decoder_layer_type)
        self.dec_layers = nn.ModuleList([
            get_transformer_decoder_layer(opt)
            for _ in range(n_layers)
        ])

    # target_seq: batch_size * len_q
    # target_pos: batch_size * len_1_q
    # fc_feats: batch_size * model_size
    # att_feats: batch_size * att_size * model_size
    # masks:  batch_size * len_1_q
    def forward(self, target_seq, target_pos, fc_feats, att_feats, masks):

        # input_dec: batch_size * len_1_q * model_size
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
        # <- target_pos: batch_size * len_1_q
        # <- masks: batch_size * len_1_q
        # -> dec_self_attn_pad_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_pad_mask = getSelfAttnPaddingMaskWithMask(target_pos, masks)

        # dec_self_attn_sub_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_sub_mask = getAttnSubsequentMask(target_pos)

        # dec_self_attn_mask: batch_size * len_1_q * len_1_q
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_sub_mask, 0)

        # output_dec: batch_size * len_1_q * model_size
        output_dec = input_dec
        for dec_layer in self.dec_layers:
            # output_dec: batch_size * len_1_q * model_size
            # att_feats: batch_size * att_size * model_size
            # dec_self_attn_mask: batch_size * len_1_q * len_1_q
            # output_dec: batch_size * len_q * model_size
            output_dec = dec_layer(output_dec, att_feats,
                                   slf_attn_mask = dec_self_attn_mask,
                                   dec_enc_attn_mask = None)

        output_dec = output_dec[:,1:,:]

        # output_dec: batch_size * len_q * model_size
        return output_dec


class ImageCMDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageCMDecoder, self).__init__()

        vocab_size = opt.vocab_size
        # words (with out EOS EOS)
        seq_length = opt.seq_length
        model_size = opt.model_size
        n_layers = opt.n_layers

        pos_size = seq_length + 2

        self.model_size = model_size

        self.pos_embed = nn.Embedding(pos_size, model_size)
        self.pos_embed.weight.data = InitPositionEcoding(pos_size, model_size)

        # batch_size * vocab_size * model_sizes
        self.word_embed = nn.Embedding(vocab_size + 1, model_size)

        # input_dec:  batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # output:     batch_size * len_q * model_size
        print('transformer_decoder_layer_type: ' + opt.transformer_decoder_layer_type)
        self.dec_layers = nn.ModuleList([
            get_transformer_decoder_layer(opt)
            for _ in range(n_layers)
        ])

    # target_seq: batch_size * len_q
    # target_pos: batch_size * len_q
    # fc_feats: batch_size * model_size
    # att_feats: batch_size * len_q * model_size
    # masks: batch_size * len_q
    def forward(self, target_seq, target_pos, fc_feats, att_feats, masks):

        batch_size, len_q = target_seq.size()

        # input_dec: batch_size * len_q * model_size
        new_masks = masks.unsqueeze(2).repeat(1, 1, self.model_size)

        # input_dec: batch_size * len_q * model_size
        input_dec = self.word_embed(target_seq)

        # fc_feats: batch_size * len_q * model_size
        fc_feats = fc_feats.unsqueeze(1).repeat(1, len_q, 1)

        #
        input_dec += fc_feats

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
        for dec_layer in self.dec_layers:
            # output_dec: batch_size * len_q * model_size
            # output_enc: batch_size * len_q * model_size
            # output_dec: batch_size * len_q * model_size
            # dec_self_attn_mask: batch_size * len_q * len_q
            # dec_enc_attn_pad_mask: batch_size * len_q * len_q
            output_dec = dec_layer(output_dec, att_feats,
                                   slf_attn_mask = dec_self_attn_mask,
                                   dec_enc_attn_mask = None)

        # output_dec: batch_size * len_q * model_size
        return output_dec


class ImageCMWGDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageCMWGDecoder, self).__init__()

        vocab_size = opt.vocab_size
        # words (with out EOS EOS)
        seq_length = opt.seq_length
        model_size = opt.model_size
        n_layers = opt.n_layers

        pos_size = seq_length + 2

        self.model_size = model_size

        self.pos_embed = nn.Embedding(pos_size, model_size)
        self.pos_embed.weight.data = InitPositionEcoding(pos_size, model_size)

        # batch_size * vocab_size * model_sizes
        self.word_embed = nn.Embedding(vocab_size + 1, model_size)

        # input_dec:  batch_size * len_q * model_size
        # output_enc: batch_size * len_q1 * model_size
        # output:     batch_size * len_q * model_size
        print('transformer_decoder_layer_type: ' + opt.transformer_decoder_layer_type)
        self.dec_layers = nn.ModuleList([
            get_transformer_decoder_layer(opt)
            for _ in range(n_layers)
        ])

        # proj weight
        self.proj_wg = nn.Linear(model_size, vocab_size + 1)
        init.xavier_normal(self.proj_wg.weight)


    # target_seq: batch_size * len_q
    # target_pos: batch_size * len_q
    # fc_feats: batch_size * model_size
    # att_feats: batch_size * len_q * model_size
    # masks: batch_size * len_q
    def forward(self, target_seq, target_pos, fc_feats, att_feats, masks):

        batch_size, len_q = target_seq.size()

        # input_dec: batch_size * len_q * model_size
        new_masks = masks.unsqueeze(2).repeat(1, 1, self.model_size)

        # input_dec: batch_size * len_q * model_size
        input_dec = self.word_embed(target_seq)

        # fc_feats: batch_size * len_q * model_size
        fc_feats = fc_feats.unsqueeze(1).repeat(1, len_q, 1)

        # input_dec: batch_size * len_q * model_size
        input_dec = input_dec + fc_feats

        # input_dec: batch_size * len_q * model_size
        # proj_wg: batch_size * len_q * (vocab_size + 1)
        proj_w = F.sigmoid(self.proj_wg(input_dec))

        # input_dec: batch_size * len_q * model_size
        input_dec = input_dec + self.pos_embed(target_pos)

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
        for dec_layer in self.dec_layers:
            # output_dec: batch_size * len_q * model_size
            # output_enc: batch_size * len_q * model_size
            # output_dec: batch_size * len_q * model_size
            # dec_self_attn_mask: batch_size * len_q * len_q
            # dec_enc_attn_pad_mask: batch_size * len_q * len_q
            output_dec = dec_layer(output_dec, att_feats,
                                   slf_attn_mask = dec_self_attn_mask,
                                   dec_enc_attn_mask = None)

        # output_dec: batch_size * len_q * model_size
        return output_dec, proj_w