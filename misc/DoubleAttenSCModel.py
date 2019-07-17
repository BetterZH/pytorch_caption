from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import *

import misc.Embed as Embed
import models
import rnn.LSTM as LSTM
import rnn.rnn_utils as rnn_utils


class DoubleAttenSCModel(nn.Module):
    def __init__(self, opt):
        super(DoubleAttenSCModel, self).__init__()
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
        self.batch_size = opt.batch_size * opt.seq_per_img
        self.rnn_atten = opt.rnn_atten
        self.num_layers = opt.num_layers
        self.num_parallels = opt.num_parallels
        self.sample_rate = opt.sample_rate
        self.use_linear = opt.use_linear
        self.rnn_size_list = opt.rnn_size_list
        self.gram_num = opt.gram_num

        # reviewnet
        self.use_reviewnet = opt.use_reviewnet
        if self.use_reviewnet == 1:
            self.review_length = opt.review_length
            self.review_nets = nn.ModuleList()
            for i in range(self.review_length):
                self.review_nets[i] = LSTM.LSTM_SOFT_ATT_NOX(self.rnn_size, self.att_size, self.drop_prob_lm)
            opt.att_size = self.review_length

        # LSTM
        # opt.input_encoding_size = opt.input_encoding_size * 2
        self.core = rnn_utils.get_lstm(opt)

        if self.rnn_atten == "ATT_LSTM":
            self.atten = LSTM.LSTM_ATTEN_LAYER(self.rnn_size)

        # self.vocab_size + 1 -> self.input_encoding_size
        # self.vocab_size + 1 -> self.input_encoding_size
        if self.gram_num > 0:
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                       Embed.WordEmbed(self.gram_num))
            # self.embed_tc = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
            #                            Embed.WordEmbed(self.gram_num))
            # self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
            # self.embed_tc = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        else:
            self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.embed_tc = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        if self.use_linear:
            # (batch_size * fc_feat_size) -> (batch_size * input_encoding_size)
            self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
            self.att_embed = nn.Linear(self.att_feat_size, self.input_encoding_size)

            # self.relu = nn.RReLU(inplace=True)
            self.relu = nn.ReLU()
            self.init_weight()

    def init_weight(self):
        init.xavier_normal(self.img_embed.weight)
        init.xavier_normal(self.att_embed.weight)

    def init_hidden(self, batch_size, fc_feats):
        # fc_feats (batch_size * size)
        inputs = []

        if self.rnn_type == 'LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_SET':
            for i in range(len(self.rnn_size_list)):
                for j in range(self.num_layers*2):
                    inputs.append(Variable(torch.FloatTensor(batch_size, self.rnn_size_list[i]).zero_()).cuda())
        elif self.rnn_type == 'GRU_DOUBLE_ATT_STACK_PARALLEL_DROPOUT':
            for i in range(len(self.rnn_size_list)):
                for j in range(self.num_layers):
                    inputs.append(Variable(torch.FloatTensor(batch_size, self.rnn_size_list[i]).zero_()).cuda())
        else:
            for i in range(self.num_layers*2):
                inputs.append(Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda())

        return inputs


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

    def review_model(self, fc_feats, att_feats):
        batch_size = fc_feats.size(0)
        review_hs = []
        review_state = self.init_hidden(batch_size, fc_feats)
        for i in range(self.review_length):
            # batch_size * rnn_size
            review_state = self.review_nets[i](att_feats, review_state)
            review_hs.append(review_state[-1])
        att_feats = torch.cat([_.unsqueeze(1) for _ in review_hs], 1).contiguous()
        return att_feats, review_state


    # fc_feats   batch_size * feat_size
    # att_feats  batch_size * att_size * feat_size
    def forward(self, fc_feats, att_feats, labels):

        batch_size = fc_feats.size(0)
        outputs = []
        embed_inputs = torch.LongTensor(batch_size, self.seq_length + 2).cuda()
        sample_max = 0
        temperature = 1

        if self.use_linear:
            fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        if self.rnn_atten == "ATT_LSTM":
            att_feats = self.atten(fc_feats, att_feats)

        if self.use_reviewnet == 1:
            att_feats, state = self.review_model(fc_feats, att_feats)
        else:
            state = self.init_hidden(batch_size, fc_feats)


        # xt -> (batch_size * input_encoding_size)
        for t in range(self.seq_length + 2):
            can_skip = False
            if t == 0:
                # batch_size * (feat_size*2)
                # xt = torch.cat((fc_feats, fc_feats), 1)
                # batch_size * feat_size
                xt = fc_feats
            elif t == 1:
                # zero for start
                # batch_size
                it = torch.LongTensor(batch_size).zero_()
                embed_inputs[:,t] = it

                it = Variable(it, requires_grad=False).cuda()
                # batch_size * rnn_size
                concat_temp = self.embed(it)
                # text-conditional image embedding
                # text-conditional image embedding
                text_condition = self.embed_tc(it)
                image_temp = F.softmax(fc_feats * text_condition)
                # concatenate the textual feature and the guidance
                # xt = torch.cat((concat_temp, image_temp), 1)
                xt = concat_temp + image_temp
                # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
            else:
                if random.randint(0,99) >= self.sample_rate:
                    it = labels[:,t-2].clone()
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
                    # batch_size * t
                    embed_inputs[:,t] = it.data

                    concat_temp = self.embed(it)
                    # batch_size * (t-1)
                    prev_inputs = embed_inputs[:,1:t]
                    # batch_size * (t-1) * input_encoding_size
                    prev_embeds = self.embed_tc(Variable(prev_inputs))
                    # batch_size * input_encoding_size
                    text_condition = prev_embeds.mean(1)

                    image_temp = F.softmax(fc_feats * text_condition)
                    # concatenate the textual feature and the guidance
                    # xt = torch.cat((concat_temp, image_temp), 1)
                    xt = concat_temp + image_temp
                    # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()

            if not can_skip:
                state, logprobs = self.core(xt, att_feats, state)
                if t > 0:
                    outputs.append(logprobs)

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
        embed_inputs = torch.LongTensor(batch_size, self.seq_length + 2).cuda()

        if self.use_linear:
            fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        if self.rnn_atten == "ATT_LSTM":
            att_feats = self.atten(fc_feats, att_feats)

        if self.use_reviewnet == 1:
            att_feats, state = self.review_model(fc_feats, att_feats)
        else:
            state = self.init_hidden(batch_size, fc_feats)

        for t in range(self.seq_length+2):
            if t == 0:
                # xt = torch.cat((fc_feats, fc_feats), 1)
                xt = fc_feats
            elif t == 1:
                it = torch.LongTensor(batch_size).cuda().zero_()
                embed_inputs[:,t] = it

                # batch_size * rnn_size
                concat_temp = self.embed(Variable(it))
                # text-conditional image embedding
                text_condition = self.embed_tc(Variable(it))
                image_temp = F.softmax(fc_feats * text_condition)
                # concatenate the textual feature and the guidance
                # xt = torch.cat((concat_temp, image_temp), 1)
                xt = concat_temp + image_temp
                # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
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

                # batch_size * t
                embed_inputs[:,t] = it

                concat_temp = self.embed(Variable(it))

                # batch_size * (t-1)
                prev_inputs = embed_inputs[:, 1:t]
                # batch_size * (t-1) * input_encoding_size
                prev_embeds = self.embed_tc(Variable(prev_inputs))
                # batch_size * input_encoding_size
                text_condition = prev_embeds.mean(1)

                image_temp = F.softmax(fc_feats * text_condition)
                # concatenate the textual feature and the guidance
                # xt = torch.cat((concat_temp, image_temp), 1)
                xt = concat_temp + image_temp
                # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
            if t >= 2:
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            state, logprobs = self.core(xt, att_feats, state)

        # batch_size * seq_length
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs


    def sample_beam(self, fc_feats, att_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        seq = torch.LongTensor(batch_size ,self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, self.seq_length).zero_()

        if self.use_linear:
            fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        if self.rnn_atten == "ATT_LSTM":
            att_feats = self.atten(fc_feats, att_feats)

        for k in range(batch_size):

            beam_seq = torch.LongTensor(self.seq_length, beam_size).cuda().zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).cuda().zero_()
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            beam_fc_feats = fc_feats[k:k+1].expand(beam_size, self.input_encoding_size).contiguous()
            beam_att_feats = att_feats[k:k+1].expand(beam_size, self.att_size, self.input_encoding_size).contiguous()
            embed_inputs = torch.LongTensor(beam_size, self.seq_length + 2).cuda()

            # state = self.init_hidden(beam_size, beam_fc_feats)

            if self.use_reviewnet == 1:
                beam_att_feats, state = self.review_model(beam_fc_feats, beam_att_feats)
            else:
                state = self.init_hidden(beam_size, beam_fc_feats)

            for t in range(self.seq_length+2):
                if t == 0:
                    # xt = torch.cat((beam_fc_feats, beam_fc_feats), 1)
                    xt = beam_fc_feats
                elif t == 1:
                    it = torch.LongTensor(beam_size).cuda().zero_()
                    embed_inputs[:, t] = it
                    it = Variable(it, requires_grad=False)
                    # batch_size * rnn_size
                    concat_temp = self.embed(it)
                    # text-conditional image embedding
                    text_condition = self.embed_tc(it)
                    image_temp = F.softmax(beam_fc_feats * text_condition)
                    # concatenate the textual feature and the guidance
                    # xt = torch.cat((concat_temp, image_temp), 1)
                    xt = concat_temp + image_temp
                    # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
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
                            new_state[state_ix][vix] = state[state_ix][v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t-2, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-2, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length+1:
                            done_beams.append({'seq':beam_seq[:, vix].clone(),
                                               'logps':beam_seq_logprobs[:, vix].clone(),
                                               'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t-2]
                    embed_inputs[:, t] = it
                    concat_temp = self.embed(Variable(it))
                    # batch_size * (t-1)
                    prev_inputs = embed_inputs[:, 1:t]
                    # batch_size * (t-1) * input_encoding_size
                    lookup_table_out = self.embed_tc(Variable(prev_inputs))
                    # batch_size * input_encoding_size
                    text_condition = lookup_table_out.mean(1)

                    image_temp = F.softmax(beam_fc_feats * text_condition)
                    # concatenate the textual feature and the guidance
                    # xt = torch.cat((concat_temp, image_temp), 1)
                    xt = concat_temp + image_temp
                    # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
                if t >= 2:
                    state = new_state

                state, logprobs = self.core(xt, beam_att_feats, state)


            done_beams = sorted(done_beams, key=lambda x: -x['p'])

            seq[k,:] = done_beams[0]['seq']
            seqLogprobs[k,:] = done_beams[0]['logps']

        return seq, seqLogprobs


    def sample(self, fc_feats, att_feats, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if sample_max == 1 and beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []
        embed_inputs = torch.LongTensor(batch_size, self.seq_length + 2).cuda()

        if self.use_linear:
            fc_feats, att_feats = self.embed_feats(fc_feats, att_feats)

        if self.rnn_atten == "ATT_LSTM":
            att_feats = self.atten(fc_feats, att_feats)

        if self.use_reviewnet == 1:
            att_feats, state = self.review_model(fc_feats, att_feats)
        else:
            state = self.init_hidden(batch_size, fc_feats)

        for t in range(self.seq_length+2):
            if t == 0:
                # xt = torch.cat((fc_feats, fc_feats), 1)
                xt = fc_feats
            elif t == 1:
                it = torch.LongTensor(batch_size).cuda().zero_()
                embed_inputs[:,t] = it

                # batch_size * rnn_size
                concat_temp = self.embed(Variable(it))
                # text-conditional image embedding
                text_condition = self.embed_tc(Variable(it))
                image_temp = F.softmax(fc_feats * text_condition)
                # concatenate the textual feature and the guidance
                # xt = torch.cat((concat_temp, image_temp), 1)
                xt = concat_temp + image_temp
                # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
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

                # batch_size * t
                embed_inputs[:,t] = it

                concat_temp = self.embed(Variable(it))

                # batch_size * (t-1)
                prev_inputs = embed_inputs[:, 1:t]
                # batch_size * (t-1) * input_encoding_size
                prev_embeds = self.embed_tc(Variable(prev_inputs))
                # batch_size * input_encoding_size
                text_condition = prev_embeds.mean(1)

                image_temp = F.softmax(fc_feats * text_condition)
                # concatenate the textual feature and the guidance
                # xt = torch.cat((concat_temp, image_temp), 1)
                xt = concat_temp + image_temp
                # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
            if t >= 2:
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            state, logprobs = self.core(xt, att_feats, state)

        # batch_size * seq_length
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs



