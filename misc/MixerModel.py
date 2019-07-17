from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import rnn.LSTM as LSTM
import rnn.rnn_utils as rnn_utils


# 1. self.core has no clone to share parameters
# 2. how to use the reinforce learning
# 3. first steps use xent
# 4. last steps use reinforce learning
class MixerModel(nn.Module):
    def __init__(self, opt):
        super(MixerModel, self).__init__()
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
        self.batch_size = 80

        # LSTM
        self.core = LSTM.LSTM_DOUBLE_ATT_TOP(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.att_size, dropout=self.drop_prob_lm)

        # self.vocab_size + 1 -> self.input_encoding_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        # (batch_size * fc_feat_size) -> (batch_size * input_encoding_size)
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.att_embed = nn.Linear(self.att_feat_size, self.input_encoding_size)

        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, fc_feats):
        # fc_feats (batch_size * size)
        return (Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda(),
                Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda())

    def embed_feats(self, fc_feats, att_feats):

        # batch_size * input_encoding_size <- batch_size * fc_feat_size
        fc_feats = F.relu(self.img_embed(fc_feats))

        # (batch_size * att_size) * att_feat_size
        att_feats = att_feats.view(-1, self.att_feat_size)
        # (batch_size * att_size) * input_encoding_size
        att_feats = F.relu(self.att_embed(att_feats))
        # batch_size * att_size * input_encoding_size
        att_feats = att_feats.view(-1, self.att_size, self.input_encoding_size)

        return fc_feats, att_feats


    # fc_feats   batch_size * feat_size
    # att_feats  batch_size * att_size * feat_size
    def forward(self, fc_feats, att_feats, seq, is_reinforce):

        batch_size = fc_feats.size(0)
        outputs = []

        fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        # xt -> (batch_size * input_encoding_size)
        for t in range(self.seq_length + 2):
            can_skip = False
            if t == 0:
                xt = fc_feats
                prev_h, prev_c = self.init_hidden(batch_size, xt)
            elif t == 1:
                # zero for start
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)
            else:
                if is_reinforce:
                    prob_prev = torch.exp(logprobs)
                    it = torch.multinomial(prob_prev, 1)
                    it = it.view(-1)
                else:
                    it = seq[:, t - 2].clone()

                if it.data.sum() == 0:
                    can_skip = True
                if not can_skip:
                    xt = self.embed(it)

            if not can_skip:
                prev_h, prev_c, logprobs, top_h = self.core(xt, att_feats, prev_h, prev_c)
                outputs.append(logprobs)

        # batch_size * (seq_length + 2) * (vocab_size + 1)
        # img ->
        # start ->
        # end ->
        batch_outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

        return batch_outputs


    def sample(self, fc_feats, att_feats, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if sample_max == 1 and beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []

        fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        for t in range(self.seq_length+2):
            if t == 0:
                xt = fc_feats
                prev_h, prev_c = self.init_hidden(batch_size, xt)
            elif t == 1:
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)
            else:
                if sample_max == 1:
                    # sampleLogprobs -> current sample logprobs
                    # it -> current index
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it)
                    it = it.view(-1)

                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)

            if t >= 2:
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            prev_h, prev_c, logprobs, top_h = self.core(xt, att_feats, prev_h, prev_c)
            # batch_size * (vocab_size + 1)

        # batch_size * seq_length * (vocab_size + 1)
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs


    def sample_beam(self, fc_feats, att_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        for k in range(batch_size):

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            for t in range(self.seq_length+2):
                if t == 0:
                    xt = fc_feats[k:k+1].expend(beam_size, self.input_encoding_size)
                    prev_h, prev_c = self.init_hidden(beam_size, xt)
                    xt_att = att_feats[k:k+1].expend(beam_size, self.input_encoding_size)
                elif t == 1:
                    it = torch.LongTensor(beam_size).zero_()
                    it = Variable(it, requires_grad=False)
                    xt = self.embed(it)
                else:
                    logprobsf = logprobs.float()
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 2:
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            local_logprob = ys[q,c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c':ix.data[q,c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x:-x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 2:
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > 2:
                            beam_seq[:t-2,vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        for state_ix in range(len(new_state)):
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t-2, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-2, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == self.vocab_size or t == self.seq_length+1:
                            done_beams.append({'seq':beam_seq[:, vix].clone(),
                                               'logps':beam_seq_logprobs[:, vix].clone(),
                                               'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t-2]
                    xt = self.embed(Variable(it.cuda()))

                if t >= 2:
                    state = new_state

                prev_h, prev_c, logprobs, top_h = self.core(xt, att_feats, prev_h, prev_c)

            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seq[:,k] = done_beams[0]['seq']
            seqLogprobs[:,k] = done_beams[0]['logps']

        return seq.transpose(0,1), seqLogprobs.transpose(0,1)




