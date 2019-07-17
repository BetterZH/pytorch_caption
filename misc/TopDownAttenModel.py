from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import *
import models
import rnn.rnn_utils as rnn_utils
import misc.Embed as Embed
import torch.nn.functional as F

class TopDownAttenModel(nn.Module):
    def __init__(self, opt):
        super(TopDownAttenModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.bu_feat_size = opt.bu_feat_size
        self.bu_size = opt.bu_size

        # LSTM
        self.core = rnn_utils.get_lstm(opt)

        # self.vocab_size + 1 -> self.input_encoding_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.img_embed = nn.Linear(self.fc_feat_size, self.rnn_size)
        self.att_embed = nn.Linear(self.bu_feat_size, self.rnn_size)

        # self.relu = nn.RReLU(inplace=True)
        self.relu = nn.ReLU()
        self.init_weight()


    def init_weight(self):
        init.xavier_normal(self.img_embed.weight)
        init.xavier_normal(self.att_embed.weight)

    def init_hidden(self, batch_size, fc_feats):
        inputs = []
        for i in range(self.num_layers * 2):
            inputs.append(Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda())
        return inputs

    def embed_feats(self, fc_feats, bu_feats):

        # batch_size * input_encoding_size <- batch_size * fc_feat_size
        fc_feats = self.relu(self.img_embed(fc_feats))

        # (batch_size * att_size) * bu_feat_size
        bu_feats = bu_feats.view(-1, self.bu_feat_size)
        # (batch_size * att_size) * input_encoding_size
        bu_feats = self.relu(self.att_embed(bu_feats))
        # batch_size * att_size * input_encoding_size
        bu_feats = bu_feats.view(-1, self.bu_size, self.input_encoding_size)

        return fc_feats, bu_feats

    def get_fc_feats(self, bu_feats):
        batch_size = bu_feats.size(0)
        bu_size = bu_feats.size(1)
        bu_feat_size = bu_feats.size(2)
        fc_feats = bu_feats.mean(1).view(batch_size, bu_feat_size)
        return fc_feats

    # att_feats:  batch_size * bu_size * bu_feat_size
    def forward(self, bu_feats, labels):

        batch_size = bu_feats.size(0)

        outputs = []

        fc_feats = self.get_fc_feats(bu_feats)

        fc_feats, bu_feats = self.embed_feats(fc_feats, bu_feats)

        state = self.init_hidden(batch_size, fc_feats)

        # xt -> (batch_size * input_encoding_size)
        for t in range(self.seq_length + 1):
            can_skip = False
            if t == 0:
                # zero for start
                it = torch.LongTensor(batch_size).cuda().zero_()
                wt = self.embed(Variable(it))
            else:
                it = labels[:,t-1]
                if it.data.sum() == 0:
                    can_skip = True
                    break
                wt = self.embed(it)

            prev_h = state[-1]
            xt = prev_h + fc_feats + wt

            if not can_skip:
                state, logprobs = self.core(xt, bu_feats, state)
                outputs.append(logprobs)

        # batch_size * (seq_length + 2) * (vocab_size + 1)
        batch_outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

        return batch_outputs


    def sample(self, bu_feats, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if sample_max == 1 and beam_size > 1:
            return self.sample_beam(bu_feats, opt)

        fc_feats = self.get_fc_feats(bu_feats)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []

        fc_feats, bu_feats = self.embed_feats(fc_feats, bu_feats)

        state = self.init_hidden(batch_size, fc_feats)

        for t in range(self.seq_length + 1):
            if t == 0:
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                wt = self.embed(it)
            else:
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
                    sampleLogprobs = logprobs.gather(1, it)
                    it = it.view(-1).long()

                it = Variable(it, requires_grad=False).cuda()
                wt = self.embed(it)

            prev_h = state[-1]
            xt = prev_h + fc_feats + wt

            if t >= 1:
                seq.append(it.data)
                seqLogprobs.append(sampleLogprobs.view(-1))

            state, logprobs = self.core(xt, bu_feats, state)
            # batch_size * (vocab_size + 1)

        # batch_size * seq_length * (vocab_size + 1)
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs


    def sample_beam(self, bu_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = bu_feats.size(0)

        fc_feats = self.get_fc_feats(bu_feats)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        seq = torch.LongTensor(batch_size, self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, self.seq_length).zero_()

        fc_feats, bu_feats = self.embed_feats(fc_feats, bu_feats)

        for k in range(batch_size):

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # beam_size
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            # beam_size * input_encoding_size
            beam_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.input_encoding_size).contiguous()

            # beam_size * att_size * input_encoding_size
            beam_bu_feats = bu_feats[k:k + 1].expand(beam_size, self.bu_size, self.input_encoding_size).contiguous()

            state = self.init_hidden(beam_size, beam_fc_feats)

            for t in range(self.seq_length + 1):
                if t == 0:
                    it = torch.LongTensor(beam_size).cuda().zero_()
                    wt = self.embed(Variable(it))
                else:
                    logprobsf = logprobs.float()
                    # beam_size * (vocab_size + 1)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    # beam_size
                    cols = min(beam_size, ys.size(1))
                    # beam_size
                    rows = beam_size
                    if t == 1:
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            local_logprob = ys[q,c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c':ix.data[q,c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x:-x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]

                    st = 1
                    if t > st:
                        beam_seq_prev = beam_seq[:t-st].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-st].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > st:
                            beam_seq[:t-st, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-st, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # vix is batch_size index
                        for state_ix in range(len(new_state)):
                            new_state[state_ix][vix] = state[state_ix][v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t-st, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-st, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        # 0 stand for the sentence start
                        if v['c'] == 0 or t == self.seq_length:
                            done_beams.append({'seq':beam_seq[:, vix].clone(),
                                               'logps':beam_seq_logprobs[:, vix].clone(),
                                               'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t-st]
                    wt = self.embed(Variable(it.cuda()))

                if t >= 1:
                    state = new_state

                prev_h = state[-1]
                xt = prev_h + beam_fc_feats + wt

                state, logprobs = self.core(xt, beam_bu_feats, state)

            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seq[k,:] = done_beams[0]['seq']
            seqLogprobs[k,:] = done_beams[0]['logps']

        return seq, seqLogprobs




