from transformer.Transformer import *
import transformer.transformer_utils as transformer_utils

class ImageAttentionDecoder(nn.Module):
    def __init__(self, opt):
        super(ImageAttentionDecoder, self).__init__()

        vocab_size = opt.vocab_size
        # words (with out BOS EOS)
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
        self.dec_layers = nn.ModuleList([
            transformer_utils.get_transformer_decoder_layer(opt)
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


