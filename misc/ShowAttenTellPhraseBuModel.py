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
import layers.GatedLayer as GatedLayer
from layers.CompactBilinearPooling import CompactBilinearPooling
import random
class ShowAttenTellPhraseBuModel(nn.Module):
    def __init__(self, opt):
        super(ShowAttenTellPhraseBuModel, self).__init__()

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
        self.word_gram_num = opt.word_gram_num
        self.phrase_gram_num = opt.phrase_gram_num
        self.conv_gram_num = opt.conv_gram_num
        self.context_len = opt.context_len
        self.use_prob_weight = opt.use_prob_weight
        self.phrase_type = opt.phrase_type
        self.mil_type = opt.mil_type
        self.use_gated_layer = getattr(opt, 'use_gated_layer', 0)

        self.sample_rate = getattr(opt, 'sample_rate', 0)


        self.word_embedding_type = getattr(opt, 'word_embedding_type', 0)

        self.bu_size = getattr(opt, 'bu_size', opt.att_size)
        self.bu_feat_size = getattr(opt, 'bu_feat_size', opt.att_feat_size)

        self.use_bilinear = getattr(opt, 'use_bilinear', False)
        self.bilinear_output = getattr(opt, 'bilinear_output', 1000)

        self.relu_type = getattr(opt, 'relu_type', 0)


        # LSTM
        self.core = rnn_utils.get_lstm(opt)

        # self.vocab_size + 1 -> self.input_encoding_size
        if self.word_embedding_type == 1:
            self.embed = Embed.EmbeddingWithBias(self.vocab_size + 1, self.input_encoding_size)
        else:
            if self.word_gram_num > 0:
                self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                           Embed.WordEmbed(self.word_gram_num))
            else:
                self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)


        # phrase embed
        if self.phrase_type == 1:
            self.phraseEmbed = Embed.PhraseEmbed(self.phrase_gram_num, self.rnn_size)
        elif self.phrase_type == 2:
            self.phraseEmbed = Embed.ConvEmbed(self.conv_gram_num)
        elif self.phrase_type == 3:
            self.phraseEmbed = Embed.PhraseEmbed(self.phrase_gram_num, self.rnn_size)
            self.phraseEmbed1 = Embed.ConvEmbed(self.conv_gram_num)

        # word weight linear
        # input_encoding_size
        if self.use_prob_weight:
            self.prob_weight_layer = nn.Sequential(nn.Linear(self.fc_feat_size, self.vocab_size + 1),
                                         nn.Softmax())

        if self.use_linear:
            self.img_embed = nn.Linear(self.fc_feat_size, self.rnn_size)
            self.att_embed = nn.Linear(self.att_feat_size, self.rnn_size)
            self.bu_embed = nn.Linear(self.bu_feat_size, self.rnn_size)

            # self.relu = nn.RReLU(inplace=True)

            if self.relu_type == 0:
                if self.use_gated_layer == 1:
                    self.relu = GatedLayer.GatedTanh(self.input_encoding_size)
                else:
                    self.relu = nn.PReLU()
            elif self.relu_type == 1:
                self.img_relu = nn.PReLU()
                self.att_relu = nn.PReLU()
                self.bu_relu = nn.PReLU()

            self.init_weight()

        if self.use_bilinear:
            self.bilinear_layer = CompactBilinearPooling(self.rnn_size, self.rnn_size, self.bilinear_output)
            self.bilinear_layer1 = nn.Linear(self.bilinear_output, self.rnn_size)

    def init_weight(self):
        init.xavier_normal(self.img_embed.weight)
        init.xavier_normal(self.att_embed.weight)
        init.xavier_normal(self.bu_embed.weight)


    def init_hidden(self, batch_size, fc_feats):

        inputs = []

        for i in range(self.num_layers * 2):
            inputs.append(Variable(torch.FloatTensor(batch_size, self.rnn_size).copy_(fc_feats.data)).cuda())

        return inputs


    def embed_feats(self, fc_feats, att_feats, bu_feats):

        # batch_size * input_encoding_size <- batch_size * fc_feat_size
        if self.relu_type == 0:
            fc_feats = self.relu(self.img_embed(fc_feats))
        elif self.relu_type == 1:
            fc_feats = self.img_relu(self.img_embed(fc_feats))

        # (batch_size * att_size) * att_feat_size
        att_feats = att_feats.view(-1, self.att_feat_size)
        # (batch_size * att_size) * input_encoding_size
        if self.relu_type == 0:
            att_feats = self.relu(self.att_embed(att_feats))
        elif self.relu_type == 1:
            att_feats = self.att_relu(self.att_embed(att_feats))
        # batch_size * att_size * input_encoding_size
        att_feats = att_feats.view(-1, self.att_size, self.input_encoding_size)

        # (batch_size * bu_size) * bu_feat_size
        bu_feats = bu_feats.view(-1, self.bu_feat_size)
        # (batch_size * att_size) * input_encoding_size
        if self.relu_type == 0:
            bu_feats = self.relu(self.bu_embed(bu_feats))
        elif self.relu_type == 1:
            bu_feats = self.bu_relu(self.bu_embed(bu_feats))
        # batch_size * att_size * input_encoding_size
        bu_feats = bu_feats.view(-1, self.bu_size, self.input_encoding_size)

        return fc_feats, att_feats, bu_feats


    # fc_feats   batch_size * feat_size
    # att_feats  batch_size * att_size * feat_size
    # bu_feats   batch_size * bu_size * bu_feat_size
    def forward(self, fc_feats, att_feats, bu_feats, labels):

        sample_max = 1
        temperature = 1.0

        batch_size = fc_feats.size(0)
        outputs = []

        if self.use_prob_weight:
            prob_weight = self.prob_weight_layer(fc_feats)

        if self.use_linear:
            fc_feats, att_feats, bu_feats = self.embed_feats(fc_feats, att_feats, bu_feats)

        state = self.init_hidden(batch_size, fc_feats)

        if self.phrase_type > 0:
            embed_inputs = torch.FloatTensor(batch_size, self.seq_length+self.context_len-1, self.rnn_size).cuda().zero_()

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

                if random.randint(0,99) >= self.sample_rate:
                    it = labels[:, t - 2].clone()
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
                xt = self.embed(it)

                if self.phrase_type == 3:
                    # t - 2 + 3
                    embed_inputs[ :,t-2+self.context_len-1] = xt.data
                    embed_inputs_v = Variable(embed_inputs[:,t-2:t-2+self.context_len])
                    # t - 2 : t - 2 + 4
                    phrase_it = self.phraseEmbed(embed_inputs_v)
                    phrase_it1 = self.phraseEmbed1(embed_inputs_v)
                    xt = xt + phrase_it + phrase_it1
                elif self.phrase_type > 0:
                    # t - 2 + 3
                    embed_inputs[:,t-2+self.context_len-1] = xt.data
                    # t - 2 : t - 2 + 4
                    phrase_it = self.phraseEmbed(Variable(embed_inputs[:,t-2:t-2+self.context_len]))
                    xt += phrase_it

                if self.use_bilinear:
                    xt = self.bilinear_layer(xt, fc_feats)
                    xt = self.bilinear_layer1(xt)

            if not can_skip:
                state, logprobs, prob_w = self.core(xt, att_feats, bu_feats, state)
                if self.use_prob_weight:
                    logprobs = logprobs * prob_weight
                if t > 0:
                    outputs.append(logprobs)

        # batch_size * (seq_length + 2) * (vocab_size + 1)
        batch_outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

        return batch_outputs, prob_w


    def sample(self, fc_feats, att_feats, bu_feats, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if sample_max == 1 and beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, bu_feats, opt)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []

        if self.use_prob_weight:
            prob_weight = self.prob_weight_layer(fc_feats)

        if self.use_linear:
            fc_feats, att_feats, bu_feats = self.embed_feats(fc_feats, att_feats, bu_feats)

        state = self.init_hidden(batch_size, fc_feats)

        if self.phrase_type > 0:
            embed_inputs = torch.FloatTensor(batch_size, self.seq_length+self.context_len-1, self.rnn_size).cuda().zero_()

        for t in range(self.seq_length+2):
            if t == 0:
                xt = fc_feats
            elif t == 1:
                it = torch.LongTensor(batch_size).zero_()
                it = Variable(it, requires_grad=False).cuda()
                xt = self.embed(it)
            else:
                it = Variable(it)
                xt = self.embed(it)

                if self.phrase_type == 3:
                    # t - 2 + 3
                    embed_inputs[:, t - 2 + self.context_len - 1] = xt.data
                    embed_inputs_v = Variable(embed_inputs[:, t - 2:t - 2 + self.context_len])
                    # t - 2 : t - 2 + 4
                    phrase_it = self.phraseEmbed(embed_inputs_v)
                    phrase_it1 = self.phraseEmbed1(embed_inputs_v)
                    xt = xt + phrase_it + phrase_it1
                elif self.phrase_type > 0:
                    # t - 2 + 3
                    embed_inputs[:,t-2+self.context_len-1] = xt.data
                    # t - 2 : t - 2 + 4
                    phrase_it = self.phraseEmbed(Variable(embed_inputs[:,t-2:t-2+self.context_len]))
                    xt += phrase_it

                if self.use_bilinear:
                    xt = self.bilinear_layer(xt, fc_feats)
                    xt = self.bilinear_layer1(xt)

            state, logprobs, prob_w = self.core(xt, att_feats, bu_feats, state)
            if self.use_prob_weight:
                logprobs = logprobs * prob_weight
            # batch_size * (vocab_size + 1)

            if t >= 1:
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

                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))


        # batch_size * (seq_length + 1) * 1
        batch_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        batch_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

        return batch_seq, batch_seqLogprobs


    def sample_beam(self, fc_feats, att_feats, bu_feats, opt={}):

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        seq = torch.LongTensor(batch_size, self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, self.seq_length).zero_()

        if self.phrase_type > 0:
            embed_inputs = torch.FloatTensor(beam_size, self.seq_length+self.context_len-1, self.rnn_size).cuda().zero_()

        if self.use_linear:
            fc_feats_new, att_feats_new, bu_feats_new = self.embed_feats(fc_feats, att_feats, bu_feats)

        for k in range(batch_size):

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # beam_size
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            beam_fc_feats_w = fc_feats[k:k + 1].expand(beam_size, self.fc_feat_size).contiguous()

            if self.use_prob_weight:
                # prob weight
                prob_weight = self.prob_weight_layer(beam_fc_feats_w)

            # beam_size * input_encoding_size
            beam_fc_feats = fc_feats_new[k:k + 1].expand(beam_size, self.input_encoding_size).contiguous()

            # beam_size * att_size * input_encoding_size
            beam_att_feats = att_feats_new[k:k + 1].expand(beam_size, self.att_size, self.input_encoding_size).contiguous()

            # beam_size * bu_size * input_encoding_size
            beam_bu_feats = bu_feats_new[k:k + 1].expand(beam_size, self.bu_size, self.input_encoding_size).contiguous()

            state = self.init_hidden(beam_size, beam_fc_feats)

            if self.phrase_type > 0:
                embed_inputs.zero_()

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
                    new_state = [_.clone() for _ in state]

                    if t > 2:
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > 2:
                            beam_seq[:t-2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # vix is batch_size index
                        for state_ix in range(len(new_state)):
                            new_state[state_ix][vix] = state[state_ix][v['q']]

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

                    if self.phrase_type == 3:
                        # t - 2 + 3
                        embed_inputs[:, t - 2 + self.context_len - 1] = xt.data
                        embed_inputs_v = Variable(embed_inputs[:, t - 2:t - 2 + self.context_len])
                        # t - 2 : t - 2 + 4
                        phrase_it = self.phraseEmbed(embed_inputs_v)
                        phrase_it1 = self.phraseEmbed1(embed_inputs_v)
                        xt = xt + phrase_it + phrase_it1
                    elif self.phrase_type > 0:
                        embed_inputs[:, t - 2 + self.context_len-1] = xt.data
                        phrase_it = self.phraseEmbed(Variable(embed_inputs[:, t - 2 : t - 2 + self.context_len]))
                        xt += phrase_it

                    if self.use_bilinear:
                        xt = self.bilinear_layer(xt, beam_fc_feats)
                        xt = self.bilinear_layer1(xt)

                if t >= 2:
                    state = new_state

                state, logprobs, prob_w = self.core(xt, beam_att_feats, beam_bu_feats, state)
                if self.use_prob_weight:
                    logprobs = logprobs * prob_weight

            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seq[k,:] = done_beams[0]['seq']
            seqLogprobs[k,:] = done_beams[0]['logps']

        return seq, seqLogprobs




