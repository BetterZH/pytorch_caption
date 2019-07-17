from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import *

import misc.Embed as Embed
import models
import rnn.rnn_utils as rnn_utils


class ShowTellModel(nn.Module):
    def __init__(self, opt):
        super(ShowTellModel, self).__init__()

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.use_linear = opt.use_linear
        self.gram_num = opt.gram_num


        # LSTM
        self.core = rnn_utils.get_lstm(opt)

        # self.vocab_size + 1 -> self.input_encoding_size
        if self.gram_num > 0:
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                       Embed.WordEmbed(self.gram_num))
        else:
            self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        if self.use_linear:
            self.img_embed = nn.Linear(self.fc_feat_size, self.rnn_size)

            # self.relu = nn.RReLU(inplace=True)
            self.relu = nn.ReLU()

            self.init_weight()


    def init_weight(self):
        init.xavier_normal(self.img_embed.weight)

    def init_hidden(self, batch_size, fc_feats):
        # fc_feats (batch_size * size)

        inputs = []

        for i in range(self.num_layers * 2):
            inputs.append(Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda())

        return inputs

    def embed_feats(self, fc_feats):

        # print(fc_feats.size())

        # batch_size * input_encoding_size <- batch_size * fc_feat_size
        fc_feats = self.relu(self.img_embed(fc_feats))

        return fc_feats

    # fc_feats   batch_size * feat_size
    # att_feats  batch_size * att_size * feat_size
    def forward(self, fc_feats, seq):

        batch_size = fc_feats.size(0)
        outputs = []

        if self.use_linear:
            fc_feats = self.embed_feats(fc_feats)

        state = self.init_hidden(batch_size, fc_feats)

        # xt -> (batch_size * input_encoding_size)
        for t in range(self.seq_length + 2):
            can_skip = False
            if t == 0:
                xt = fc_feats
            elif t == 1:
                # zero for start
                it = torch.LongTensor(batch_size).cuda().zero_()
                xt = self.embed(Variable(it))
            else:
                it = seq[:,t-2]
                if it.data.sum() == 0:
                    can_skip = True
                    break
                xt = self.embed(it)

            if not can_skip:
                state, logprobs = self.core(xt, state)
                if t > 0:
                    outputs.append(logprobs)

        # batch_size * (seq_length + 1) * (vocab_size + 1)
        batch_outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

        return batch_outputs


    def sample(self, fc_feats, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if sample_max == 1 and beam_size > 1:
            return self.sample_beam(fc_feats, opt)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []

        if self.use_linear:
            fc_feats = self.embed_feats(fc_feats)

        state = self.init_hidden(batch_size, fc_feats)

        for t in range(self.seq_length+2):
            if t == 0:
                xt = fc_feats
            elif t == 1:
                it = torch.LongTensor(batch_size).cuda().zero_()
                it = Variable(it, requires_grad=False)
                xt = self.embed(it)
            else:
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
                    # it: batch_size * 1
                    it = torch.multinomial(prob_prev, 1)
                    # logprobs : batch_size * (vocab_size + 1)
                    # sampleLogprobs : batch_size * 1
                    sampleLogprobs = logprobs.gather(1, it)
                    it = it.view(-1).long()

                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)

            if t >= 2:
                seq.append(it.data)
                seqLogprobs.append(sampleLogprobs.view(-1))

            # logprobs batch_size
            state, logprobs = self.core(xt, state)

        # batch_size * seq_length * (vocab_size + 1)
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs


    def sample_beam(self, fc_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        seq = torch.LongTensor(batch_size, self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, self.seq_length).zero_()

        if self.use_linear:
            fc_feats = self.embed_feats(fc_feats)

        for k in range(batch_size):

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # beam_size
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            # beam_size * input_encoding_size
            beam_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.input_encoding_size).contiguous()

            # prev_h : beam_size * rnn_size
            # prev_c : beam_size * rnn_size
            state = self.init_hidden(beam_size, beam_fc_feats)

            for t in range(self.seq_length + 2):
                if t == 0:
                    xt = beam_fc_feats
                elif t == 1:
                    it = torch.LongTensor(beam_size).cuda().zero_()
                    xt = self.embed(Variable(it))
                else:
                    logprobsf = logprobs.float()
                    # beam_size * (vocab_size + 1)
                    # ys : sorted
                    # ix : indices (vocab_size + 1)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    # beam_size
                    cols = min(beam_size, ys.size(1))
                    # beam_size
                    rows = beam_size
                    if t == 2:
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            # c : indices
                            # q : rows
                            # candidate_logprob
                            # local_logprob
                            candidates.append({'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]

                    if t > 2:
                        # t - 2 is the real word
                        beam_seq_prev = beam_seq[:t - 2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > 2:
                            # t - 2 is the real word
                            beam_seq[:t - 2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # vix is batch_size index
                        # prev_h
                        # prev_c
                        for state_ix in range(len(new_state)):
                            new_state[state_ix][vix] = state[state_ix][v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t - 2, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 2, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        # 0 stand for the sentence start
                        if v['c'] == 0 or t == self.seq_length + 1:
                            done_beams.append({'seq': beam_seq[:, vix].clone(),
                                               'logps': beam_seq_logprobs[:, vix].clone(),
                                               'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t - 2]
                    xt = self.embed(Variable(it.cuda()))

                if t >= 2:
                    state = new_state

                state, logprobs = self.core(xt, state)

            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seq[k, :] = done_beams[0]['seq']
            seqLogprobs[k, :] = done_beams[0]['logps']

        return seq, seqLogprobs




