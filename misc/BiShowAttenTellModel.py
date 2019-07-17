from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import *

import rnn.LSTM as LSTM


class BiShowAttenTellModel(nn.Module):
    def __init__(self, opt):
        super(BiShowAttenTellModel, self).__init__()
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
        self.output_size = self.vocab_size + 1

        # LSTM
        # self.core = nn.LSTM(self.input_encoding_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        if self.rnn_type == "LSTM_SOFT_ATT":
            self.core = LSTM.LSTM_SOFT_ATT_TOP(self.input_encoding_size, self.output_size, self.rnn_size, self.att_size, dropout=self.drop_prob_lm)
            self.core1 = LSTM.LSTM_SOFT_ATT_TOP(self.input_encoding_size, self.output_size, self.rnn_size, self.att_size, dropout=self.drop_prob_lm)
        elif self.rnn_type == "LSTM_DOUBLE_ATT":
            self.core = LSTM.LSTM_DOUBLE_ATT_TOP(self.input_encoding_size, self.output_size, self.rnn_size, self.att_size, dropout=self.drop_prob_lm)
            self.core1 = LSTM.LSTM_DOUBLE_ATT_TOP(self.input_encoding_size, self.output_size, self.rnn_size, self.att_size, dropout=self.drop_prob_lm)
        else:
            raise Exception("rnn type not supported: {}".format(self.rnn_type))

        # self.vocab_size + 1 -> self.input_encoding_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.img_embed = nn.Linear(self.fc_feat_size, self.rnn_size)
        self.att_embed = nn.Linear(self.att_feat_size, self.rnn_size)

        self.proj = nn.Linear(self.rnn_size, self.output_size)

        # self.relu = nn.RReLU(inplace=True)
        self.relu = nn.PReLU()

        self.init_weight()


    def init_weight(self):
        init.xavier_normal(self.img_embed.weight)
        init.xavier_normal(self.att_embed.weight)
        init.xavier_normal(self.proj.weight)

    def init_hidden(self, batch_size, fc_feats):
        # fc_feats (batch_size * size)
        return (Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda(),
                Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda())

    def embed_feats(self, fc_feats, att_feats):

        # batch_size * input_encoding_size <- batch_size * fc_feat_size
        fc_feats = self.relu(self.img_embed(fc_feats))

        # (batch_size * att_size) * att_feat_size
        att_feats = att_feats.view(-1, self.att_feat_size)
        # (batch_size * att_size) * input_encoding_size
        att_feats = self.relu(self.att_embed(att_feats))
        # batch_size * att_size * input_encoding_size
        att_feats = att_feats.view(-1, self.att_size, self.input_encoding_size)

        return fc_feats, att_feats


    # fc_feats   batch_size * feat_size
    # att_feats  batch_size * att_size * feat_size
    def forward(self, fc_feats, att_feats, labels):

        batch_size = fc_feats.size(0)
        htops = []
        htops1 = []

        fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        prev_h, prev_c = self.init_hidden(batch_size, fc_feats)

        max_len = self.seq_length + 2

        # xt -> (batch_size * input_encoding_size)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = fc_feats
            elif t == 1:
                # zero for start
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)
            else:
                it = labels[:,t-2].clone()
                if it.data.sum() == 0:
                    max_len = t
                    break
                xt = self.embed(it)

            prev_h, prev_c, htop = self.core(xt, att_feats, prev_h, prev_c)
            if t > 0:
                htops.append(htop)

        # backward
        prev_h, prev_c = self.init_hidden(batch_size, fc_feats)
        for t in range(max_len):
            if t == 0:
                xt = fc_feats
            elif t == 1:
                # zero for start
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)
            else:
                it = labels[:, max_len - 1 - t - 1].clone()
                xt = self.embed(it)

            prev_h, prev_c, htop = self.core1(xt, att_feats, prev_h, prev_c)
            if t > 0:
                htops1.append(htop)

        outputs = []
        for i in xrange(max_len-1):
            htop = htops[i] + htops1[max_len-1-i-1]
            output = F.log_softmax(self.proj(htop))
            outputs.append(output)

        # batch_size * (seq_length + 2) * (vocab_size + 1)
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

        prev_h, prev_c = self.init_hidden(batch_size, fc_feats)

        for t in range(self.seq_length+2):
            if t == 0:
                xt = fc_feats
            elif t == 1:
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)
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
                xt = self.embed(it)

            if t >= 2:
                seq.append(it.data)
                seqLogprobs.append(sampleLogprobs.view(-1))

            prev_h, prev_c, htop = self.core(xt, att_feats, prev_h, prev_c)

            logprobs = F.log_softmax(self.proj(htop))
            # batch_size * (vocab_size + 1)


        # batch_size * seq_length * (vocab_size + 1)
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs


    def sample_beam(self, fc_feats, att_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        seq = torch.LongTensor(batch_size, self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, self.seq_length)

        fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        for k in range(batch_size):

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # beam_size
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            # beam_size * input_encoding_size
            beam_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.input_encoding_size).contiguous()

            # beam_size * att_size * input_encoding_size
            beam_att_feats = att_feats[k:k + 1].expand(beam_size, self.att_size, self.input_encoding_size).contiguous()

            prev_h, prev_c = self.init_hidden(beam_size, beam_fc_feats)

            for t in range(self.seq_length+2):
                if t == 0:
                    xt = beam_fc_feats
                elif t == 1:
                    it = torch.LongTensor(beam_size).cuda().zero_()
                    xt = self.embed(Variable(it))
                else:
                    logprobsf = logprobs.float()
                    # beam_size * (vocab_size + 1)
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
                            local_logprob = ys[q,c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c':ix.data[q,c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x:-x['p'])

                    # construct new beams
                    new_prev_h = prev_h.clone()
                    new_prev_c = prev_c.clone()

                    if t > 2:
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > 2:
                            beam_seq[:t-2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # vix is batch_size index
                        new_prev_h[vix] = prev_h[v['q']]
                        new_prev_c[vix] = prev_c[v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t-2, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-2, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        # 0 stand for the sentence start
                        if v['c'] == 0 or t == self.seq_length+1:
                            done_beams.append({'seq':beam_seq[:, vix].clone(),
                                               'logps':beam_seq_logprobs[:, vix].clone(),
                                               'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t-2]
                    xt = self.embed(Variable(it.cuda()))

                if t >= 2:
                    prev_h = new_prev_h
                    prev_c = new_prev_c

                prev_h, prev_c, logprobs = self.core(xt, beam_att_feats, prev_h, prev_c)

            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seq[k,:] = done_beams[0]['seq']
            seqLogprobs[k,:] = done_beams[0]['logps']

        return seq, seqLogprobs




