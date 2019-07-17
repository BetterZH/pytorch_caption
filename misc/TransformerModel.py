from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import *
from transformer.Transformer import *
from transformer.Beam import *
import misc.utils as utils
import transformer.transformer_utils as transformer_utils

class TransformerModel(nn.Module):
    def __init__(self, opt):
        super(TransformerModel, self).__init__()

        self.model_size = opt.model_size
        self.vocab_size = opt.vocab_size
        # the number of the words in the sentence (exclude the start token and end token)
        self.seq_length = opt.seq_length
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_feat_size = opt.att_feat_size
        self.n_best = opt.n_best
        self.vocab = opt.vocab
        self.is_show_result = opt.is_show_result
        self.image_embed_type = getattr(opt, 'image_embed_type', 0)

        # Transformer
        self.head_size = opt.head_size
        self.n_layers = opt.n_layers
        self.k_size = opt.k_size
        self.v_size = opt.v_size
        self.inner_layer_size = opt.inner_layer_size

        # batch_size * att_size * att_feat_size
        linear_image = nn.Linear(self.att_feat_size, self.model_size)
        init.xavier_normal(linear_image.weight)
        if self.image_embed_type == 0:
            self.image_embed = linear_image
        else:
            self.image_embed = nn.Sequential(linear_image,
                                             nn.ReLU())

        # batch_size * len_q * model_size
        # vocab_size, seq_len, head_size=8, model_size=512,
        #         n_layers=6, k_size=64, v_size=64, inner_layer_size=2048
        self.decoder = transformer_utils.get_transformer_decoder(opt)

        # <- batch_size * len_q * model_size
        # -> batch_size * len_q * vocab_size
        self.proj = nn.Linear(self.model_size, self.vocab_size + 1)

        # share weight
        # word_embed: batch_size * vocab_size * model_size
        # self.proj.linear.weight = self.decoder.word_embed.weight


    def get_trainable_parameters(self):
        # enc_freezed_param_ids = set(map(id, self.encoder.pos_embed.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.pos_embed.parameters()))
        # freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        freezed_param_ids = dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)


    # def get_trainable_parameters(self):
    #     return self.parameters()

    # att_feats:  batch_size * att_size * att_feat_size
    # target_seq: batch_size * len_q
    def forward(self, att_feats, target_seq, masks):

        # sents = utils.decode_sequence(self.vocab, target_seq.data[:,1:])
        # print("================target_seq================")
        # for k, sent in enumerate(sents):
        #     print(sent)
        # print("================target_seq================")

        batch_size, att_size, att_feat_size = att_feats.size()
        batch_size, len_q = target_seq.size()

        # input_pos: batch_size * len_q
        # input_pos = np.array([
        #     [pos_i + 1 for pos_i in range(att_size)]
        #     for i in range(batch_size)])
        # input_pos = torch.LongTensor(input_pos).cuda()
        # input_pos = Variable(input_pos, requires_grad=False)

        # target_pos: batch_size * len_q
        target_pos = np.array([
            [j + 1 for j in range(len_q)]
            for _ in range(batch_size)])
        target_pos = torch.LongTensor(target_pos).cuda()
        target_pos = Variable(target_pos, requires_grad=False)
        target_pos = target_pos * masks.long()

        output_enc = self.image_embed(att_feats)

        # target_seq: batch_size * len_q
        # target_pos: batch_size * len_q
        # input_seq:  batch_size * len_q
        # output_enc: batch_size * len_q * model_size
        # output_dec: batch_size * len_q * model_size
        # masks: batch_size * len_q
        output_dec = self.decoder(target_seq, target_pos, output_enc, masks)

        if self.drop_prob_lm > 0:
            output_dec = F.dropout(output_dec, self.drop_prob_lm)

        # output_dec:   batch_size * len_q * model_size
        # seq_logsofts: batch_size * len_q * output_size
        seq_logsofts = F.log_softmax(self.proj(output_dec), -1)


        # sents = utils.decode_sequence(self.vocab, target_seq.data[:,1:])
        # print("================target_seq================")
        # for k, sent in enumerate(sents):
        #     print(sent)
        # print("================target_seq================")

        if self.is_show_result:
            sampleLogprobs, seq = torch.max(seq_logsofts.data, 2)
            sents = utils.decode_sequence(self.vocab, seq)
            target_sents = utils.decode_sequence(self.vocab, target_seq.data[:, 1:])
            print("===============output=================")
            for k, sent in enumerate(sents):
                print(sent," | ",target_sents[k])
            print("===============output=================")


        # seq_logsofts: batch_size * len_q * output_size
        return seq_logsofts

    def sample_rl(self, att_feats, target_seq, masks, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        # sents = utils.decode_sequence(self.vocab, target_seq.data[:,1:])
        # print("================target_seq================")
        # for k, sent in enumerate(sents):
        #     print(sent)
        # print("================target_seq================")

        batch_size, att_size, att_feat_size = att_feats.size()
        batch_size, len_q = target_seq.size()

        # input_pos: batch_size * len_q
        # input_pos = np.array([
        #     [pos_i + 1 for pos_i in range(att_size)]
        #     for i in range(batch_size)])
        # input_pos = torch.LongTensor(input_pos).cuda()
        # input_pos = Variable(input_pos, requires_grad=False)

        # target_pos: batch_size * len_q
        target_pos = np.array([
            [j + 1 for j in range(len_q)]
            for _ in range(batch_size)])
        target_pos = torch.LongTensor(target_pos).cuda()
        target_pos = Variable(target_pos, requires_grad=False)
        target_pos = target_pos * masks.long()

        output_enc = self.image_embed(att_feats)

        # target_seq: batch_size * len_q
        # target_pos: batch_size * len_q
        # input_seq:  batch_size * len_q
        # output_enc: batch_size * len_q * model_size
        # output_dec: batch_size * len_q * model_size
        # masks: batch_size * len_q
        output_dec = self.decoder(target_seq, target_pos, output_enc, masks)

        if self.drop_prob_lm > 0:
            output_dec = F.dropout(output_dec, self.drop_prob_lm)

        # output_dec:   batch_size * len_q * model_size
        # seq_logsofts: batch_size * len_q * output_size
        seq_logsofts = F.log_softmax(self.proj(output_dec), -1)

        # greedy
        # greedyLogprobs: batch_size * len_q
        # greedySeq : batch_size * len_q
        greedyLogprobs, greedySeq = torch.max(seq_logsofts.data, -1)

        # sample
        # seq_logsofts: batch_size * len_q * output_size
        if temperature == 1.0:
            prob_prev = torch.exp(seq_logsofts)
        else:
            prob_prev = torch.exp(torch.div(seq_logsofts, temperature))

        prob_size = prob_prev.size()
        # (batch_size * len_q) * output_size
        prob_prev = prob_prev.view(-1, prob_size[-1])
        # it: (batch_size * len_q) * 1
        it = torch.multinomial(prob_prev, 1)
        # it: batch_size * len_q * 1
        it = it.view(prob_size[0], prob_size[1], 1)

        # batch_size * len_q * 1
        sampleLogprobs = seq_logsofts.gather(-1, it).view(prob_size[:-1])
        sampleSeq = it.data.view(prob_size[:-1])

        # greedySeq: batch_size * len_q
        # greedyLogprobs: batch_size * len_q
        # sampleSeq: batch_size * len_q
        # sampleLogprobs: batch_size * len_q
        return greedySeq, greedyLogprobs, sampleSeq, sampleLogprobs


    # att_feats:  batch_size * att_size * att_feat_size
    def sample(self, att_feats, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        if beam_size > 1:
            return self.sample_beam(att_feats, opt)

        batch_size = att_feats.size(0)

        # output_enc: batch_size * len_q1 * model_size
        output_enc = self.image_embed(att_feats)

        masks = [torch.LongTensor(batch_size).fill_(1).cuda()]
        seq = [torch.LongTensor(batch_size).fill_(0).cuda()]
        seqLogprobs = []

        for i in range(self.seq_length + 1):

            len_dec_seq = i + 1

            # batch_size * len_dec_seq
            masks_partial_seq = torch.cat([_.unsqueeze(1) for _ in masks], 1)
            masks_partial_seq = Variable(masks_partial_seq, requires_grad=False)

            # batch_size * len_dec_seq
            dec_partial_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
            dec_partial_seq = Variable(dec_partial_seq, requires_grad=False)

            # target_pos: batch_size * len_dec_seq
            dec_partial_pos = np.array([
                [j + 1 for j in range(len_dec_seq)]
                for _ in range(batch_size)])
            dec_partial_pos = torch.LongTensor(dec_partial_pos).cuda()
            dec_partial_pos = Variable(dec_partial_pos, requires_grad=False)
            dec_partial_pos = dec_partial_pos * masks_partial_seq.long()

            # dec_partial_seq: batch_size * len_dec_seq
            # dec_partial_pos: batch_size * len_dec_seq
            # output_enc: batch_size * len_q1 * model_size
            # output_dec: batch_size * len_dec_seq * model_size
            output_dec = self.decoder(dec_partial_seq, dec_partial_pos, output_enc, masks_partial_seq.float())

            # output_dec: batch_size * model_size
            output_dec = output_dec[:, -1, :]

            if self.drop_prob_lm > 0:
                output_dec = F.dropout(output_dec, self.drop_prob_lm)

            # output_dec:   batch_size * model_size
            logprobs = F.log_softmax(self.proj(output_dec), -1)

            # it stand for the prev result
            if sample_max == 1:
                # sampleLogprobs -> current sample logprobs
                # it -> current index
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, Variable(it))
                it = it.view(-1).long()

            if i == 0:
                masks.append(torch.ones(batch_size).cuda().long())
            else:
                masks.append((seq[-1] > 0).long() & masks[-1].long())

            if masks[-1].sum() == 0:
                break

            seq.append(it)
            seqLogprobs.append(sampleLogprobs.view(-1))

        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq[1:]], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        if self.is_show_result:
            sents = utils.decode_sequence(self.vocab, batch_seq)
            print("===============output=================")
            for k, sent in enumerate(sents):
                print(sent)
            print("===============output=================")

        # sampleSeq: batch_size * seq_partial_length
        # sampleLogprobs: batch_size * seq_partial_length
        return batch_seq, batch_seqLogprobs


    # att_feats:  batch_size * att_size * att_feat_size
    def sample_beam(self, att_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        # output_enc: batch_size * len_q1 * model_size
        output_enc = self.image_embed(att_feats)

        # att_feats: (batch_size * beam_size) * att_size * att_feat_size
        att_feats = Variable(
            att_feats.data.repeat(1, beam_size, 1).view(
                att_feats.size(0) * beam_size, att_feats.size(1), att_feats.size(2)
            )
        )

        # output_enc: (batch_size * beam_size) * len_q1 * model_size
        output_enc = Variable(
            output_enc.data.repeat(1, beam_size, 1).view(
            output_enc.size(0) * beam_size, output_enc.size(1), output_enc.size(2)
        ))

        # Prepare beams
        beams = [Beam(beam_size) for _ in range(batch_size)]
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))
        }
        n_remaining_sents = batch_size

        # Decode
        for i in range(self.seq_length + 1):

            len_dec_seq = i + 1

            # (n_remaining_sents*beam_size) * len_dec_seq
            masks = torch.FloatTensor(n_remaining_sents * beam_size, len_dec_seq).fill_(1)
            masks = Variable(masks.cuda())

            # n_remaining_sents * beam_size * len_dec_seq
            dec_partial_seq = torch.stack([
                b.get_current_state() for b in beams if not b.done
            ])
            # (n_remaining_sents * beam_size) * len_dec_seq
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            dec_partial_seq = Variable(dec_partial_seq, volatile=True).cuda()

            # size: 1 * len_dec_seq
            dec_partial_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0)
            # size: (n_remaining_sents * beam_size) * len_dec_seq
            dec_partial_pos = dec_partial_pos.repeat(n_remaining_sents * beam_size, 1)
            dec_partial_pos = Variable(dec_partial_pos.type(torch.LongTensor), volatile=True).cuda()

            # dec_partial_seq: (n_remaining_sents * beam_size) * len_dec_seq
            # dec_partial_pos: (n_remaining_sents * beam_size) * len_dec_seq
            # output_enc: (n_remaining_sents * beam_size) * len_q1 * model_size
            # output_dec: (n_remaining_sents * beam_size) * len_dec_seq * model_size
            output_dec = self.decoder(dec_partial_seq, dec_partial_pos, output_enc, masks)

            # (n_remaining_sents * beam_size) * model_size
            output_dec = output_dec[:,-1,:]

            # (n_remaining_sents * beam_size) * (vocab_size+1)
            output = F.log_softmax(self.proj(output_dec), -1)

            # n_remaining_sents * beam_size * (vocab_size+1)
            word_lk = output.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            for beam_idx in range(batch_size):
                if beams[beam_idx].done:
                    continue

                inst_idx = beam_inst_idx_map[beam_idx]
                if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                    active_beam_idx_list += [beam_idx]

            if not active_beam_idx_list:
                # all instances have finished their path to <EOS>
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = [beam_inst_idx_map[k] for k in active_beam_idx_list]
            active_inst_idxs = torch.LongTensor(active_inst_idxs).cuda()

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

            # seq_var: (n_remaining_sents * beam_size) * att_size * att_feat_size
            def update_active_seq(seq_var, active_inst_idxs):
                ''' Remove the src sequence of finished instances in one batch. '''

                inst_idx_dim_size, rest_dim_size1, rest_dim_size2 = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, rest_dim_size1, rest_dim_size2)

                # select the active instances in batch
                # original_seq_data: n_remaining_sents * (beam_size * att_size) * att_feat_size
                original_seq_data = seq_var.data.view(n_remaining_sents, -1, self.att_feat_size)
                active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
                active_seq_data = active_seq_data.view(*new_size)

                # active_seq_data: (inst_idx_dim_size * beam_size) * att_size * att_feat_size
                return Variable(active_seq_data, volatile=True)

            # enc_info_var: (n_remaining_sents * beam_size) * len_q1 * model_size
            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                inst_idx_dim_size, rest_dim_size1, rest_dim_size2 = enc_info_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, rest_dim_size1, rest_dim_size2)

                # select the active instances in batch
                # original_enc_info_data: n_remaining_sents * (beam_size * len_q1) * model_size
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.model_size)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)

                # active_enc_info_data: (inst_idx_dim_size * beam_size) * len_q1 * model_size
                return Variable(active_enc_info_data, volatile=True)

            # att_feats: (inst_idx_dim_size * beam_size) * att_size * att_feat_size
            att_feats = update_active_seq(att_feats, active_inst_idxs)

            # output_enc: (inst_idx_dim_size * beam_size) * len_q1 * model_size
            output_enc = update_active_enc_info(output_enc, active_inst_idxs)

            # - update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        # - Return useful information
        # batch_size * len_q * n_best
        all_hyp, all_scores = [], []
        n_best = self.n_best

        for beam_idx in range(batch_size):
            scores, tail_idxs = beams[beam_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]

        seq = torch.LongTensor(batch_size, self.seq_length + 1).zero_()
        for i in range(batch_size):
            for j in range(len(all_hyp[i][0])):
                seq[i,j] = all_hyp[i][0][j]

        # batch_size * seq_len
        seqLogprobs = all_scores

        return seq, seqLogprobs
