from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import torch
import torch.nn as nn
from torch.autograd import *

import models
import rnn.rnn_utils as rnn_utils


class MoreAttenModel(nn.Module):
    def __init__(self, opt):
        super(MoreAttenModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.num_layers = opt.num_layers
        self.seq_length = opt.seq_length
        self.rnn_size = opt.rnn_size
        self.batch_size = opt.batch_size * opt.seq_per_img
        self.sample_rate = opt.sample_rate
        self.att_size = opt.att_size
        self.att_feat_size = opt.att_feat_size

        # LSTM
        self.core = rnn_utils.get_lstm(opt)

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.att_embed = nn.Linear(self.att_feat_size, self.input_encoding_size)

        self.relu = nn.ReLU()


    def embed_feats(self, att_feats):

        # (batch_size * att_size) * att_feat_size
        att_feats = att_feats.view(-1, self.att_feat_size)
        # (batch_size * att_size) * input_encoding_size
        att_feats = self.relu(self.att_embed(att_feats))
        # batch_size * att_size * input_encoding_size
        att_feats = att_feats.view(-1, self.att_size, self.input_encoding_size)

        return att_feats


    def init_hidden(self, batch_size):
        # fc_feats (batch_size * size)
        inputs = []

        for i in range(self.num_layers*2):
                inputs.append(Variable(torch.FloatTensor(batch_size, self.rnn_size).zero_()).cuda())

        return inputs



    # fc_feats   batch_size * feat_size
    # att_feats  batch_size * att_size * feat_size
    def forward(self, att_feats, labels):

        batch_size = att_feats.size(0)
        outputs = []
        sample_max = 0
        temperature = 1

        att_feats = self.embed_feats(att_feats)

        state = self.init_hidden(batch_size)

        # xt -> (batch_size * input_encoding_size)
        for t in range(self.seq_length + 1):
            can_skip = False
            if t == 0:
                # zero for start
                # batch_size
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                # batch_size * rnn_size
                xt = self.embed(it)
            else:
                if random.randint(0,99) >= self.sample_rate:
                    it = labels[:,t-1].clone()
                else:
                    if sample_max == 1:
                        # sampleLogprobs -> current sample logprobs
                        # it -> current index
                        sampleLogprobs, it = torch.max(logprobs.data, 1)
                    else:
                        if temperature == 1.0:
                            prob_prev = torch.exp(logprobs.data)
                        else:
                            prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                        it = torch.multinomial(prob_prev, 1)
                        # sampleLogprobs = logprobs.gather(1, Variable(it))

                    it = Variable(it.view(-1).long())

                if it.data.sum() == 0:
                    can_skip = True
                    break

                if not can_skip:
                    xt = self.embed(it)

            if not can_skip:
                state, logprobs = self.core(xt, att_feats, state)
                outputs.append(logprobs)

        # batch_size * (seq_length + 1) * (vocab_size + 1)
        batch_outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

        return batch_outputs


    def sample(self, att_feats, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if sample_max == 1 and beam_size > 1:
            return self.sample_beam(att_feats, opt)

        batch_size = att_feats.size(0)
        seq = []
        seqLogprobs = []

        att_feats = self.embed_feats(att_feats)

        state = self.init_hidden(batch_size)

        for t in range(self.seq_length+1):
            if t == 0:
                it = torch.LongTensor(batch_size).cuda().zero_()
                xt = self.embed(Variable(it))
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
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, Variable(it))
                    it = it.view(-1).long()

                xt = self.embed(Variable(it))

            if t >= 2:
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            state, logprobs = self.core(xt, att_feats, state)

        # batch_size * seq_length
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs


    def sample_beam(self, att_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        seq = torch.LongTensor(batch_size ,self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, self.seq_length).zero_()

        att_feats = self.embed_feats(att_feats)

        for k in range(batch_size):

            beam_seq = torch.LongTensor(self.seq_length, beam_size).cuda().zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).cuda().zero_()
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            beam_att_feats = att_feats[k:k+1].expand(beam_size, self.att_size, self.input_encoding_size).contiguous()

            state = self.init_hidden(beam_size)

            for t in range(self.seq_length + 1):
                if t == 0:
                    it = torch.LongTensor(beam_size).cuda().zero_()
                    it = Variable(it, requires_grad=False)
                    # batch_size * rnn_size
                    xt = self.embed(it)
                else:
                    logprobsf = logprobs.float()
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
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

                    if t > 1:
                        beam_seq_prev = beam_seq[:t-1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-1].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > 1:
                            beam_seq[:t-1,vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        for state_ix in range(len(new_state)):
                            new_state[state_ix][vix] = state[state_ix][v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t-1, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-1, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length:
                            done_beams.append({'seq':beam_seq[:, vix].clone(),
                                               'logps':beam_seq_logprobs[:, vix].clone(),
                                               'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t-1]
                    xt = self.embed(Variable(it))
                if t >= 1:
                    state = new_state

                state, logprobs = self.core(xt, beam_att_feats, state)


            done_beams = sorted(done_beams, key=lambda x: -x['p'])

            seq[k,:] = done_beams[0]['seq']
            seqLogprobs[k,:] = done_beams[0]['logps']

        return seq, seqLogprobs




