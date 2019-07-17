from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle

import torch
from torch.autograd import Variable

import misc.utils as utils
import models
from data.DataLoaderRaw import *
from data.DataLoaderThreadNew import *

import random
import sys
from json import encoder
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import *

import misc.Embed as Embed
import models
import rnn.LSTM as LSTM
import rnn.rnn_utils as rnn_utils
import vis.visual_rnn as visual_rnn
import numpy as np


# val2014 40504
# test2014 40775
def parse_args():
    # Input arguments and options
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--input_json', type=str, default='/media/amds/data/dataset/mscoco/data_coco.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_h5', type=str, default='/media/amds/data/dataset/mscoco/data_coco.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_anno', type=str, default='/media/amds/data/dataset/mscoco/anno_coco.json',
                        help='')
    parser.add_argument('--images_root', default='/media/amds/disk/dataset/mscoco',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    # Basic options
    parser.add_argument('--batch_size', type=int, default=1,
                    help='if > 0 then overrule, otherwise load from checkpoint.')

    # Sampling options
    parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # For evaluation on a folder of images:
    parser.add_argument('--coco_caption_path', type=str, default='/media/amds/data/caption/coco-caption',
                    help='coco_caption_path')
    parser.add_argument('--image_folder', type=str, default='/media/amds/disk/dataset/mscoco',
                    help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--start_from_best', type=str, default='/media/amds/disk/caption/ensemble/',
                        help="")
    parser.add_argument('--start_from', type=str, default='/media/amds/disk/caption/ensemble/',
                        help="")
    parser.add_argument('--output_dir', type=str, default='/media/amds/disk/caption/ensemble/',
                        help='')
    parser.add_argument('--datasets', type=list, default=['val2014','test2014'],
                        help='val2014, test2014')
    # misc
    # ['soft_att_weight_1_rl_1', 'dasc_sup_1_rl_2','dasc_sup_1_rl_3','dasc_sup_1_rl_4']

    list_coco1 = ['coco_cvpr_wordgate_2_beta_08']
    list_coco2 = 'coco_alpha_7500_beta_08','coco_alpha_15000_beta_08','coco_alpha_500_beta_08','coco_alpha_1_beta_08','coco_alpha_5000_beta_09','coco_alpha_5000_beta_07','coco_alpha_12500_beta_08','coco_alpha_17500_beta_08','coco_alpha_20000_beta_08','coco_alpha_5000_beta_10'
    parser.add_argument('--ids', type=list, default=list_coco2,
                        help='')
    parser.add_argument('--eval_type', type=int, default=0,
                        help='0 local, 1 server')
    parser.add_argument('--input_type', type=int, default=1,
                        help='0 sup, 1 weight')

    parser.add_argument('--start', type=int, default=0,
                        help='')
    parser.add_argument('--num', type=int, default=-1,
                        help='-1 total')

    return parser.parse_args()


def language_eval(preds, coco_caption_path, input_anno):

    sys.path.append(coco_caption_path)
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # annFile = os.path.join(coco_caption_path, 'annotations/captions_val2014.json')

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    random.seed(time.time())
    # tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    tmp_name = "temp_ensamble"

    coco = COCO(input_anno)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(tmp_name + '.json', 'w'))  # serialize to temporary json file. Sigh, COCO API...

    resFile = tmp_name + '.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
    os.system('rm ' + tmp_name + '.json')

    # create output dictionary
    out = {}
    out_str = []

    # for metric, score in cocoEval.eval.items():
    #     out[metric] = score
    #     out_str.append(str(score))

    metrics = ["CIDEr", "ROUGE_L", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    for metric in metrics:
        score = cocoEval.eval[metric]
        out[metric] = score
        out_str.append(str(score))
    str_result = ','.join(out_str)

    return out, str_result


def save_result(output_dir, str_stats, predictions):
    eval_result_file = os.path.join(output_dir, "_ensemble.csv")
    with open(eval_result_file, 'a') as f:
        f.write(str_stats + "\n")

    predictions_file = os.path.join(output_dir, "_ensemble.json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

def format_time(left_time):

    if left_time > 3600:
        left_h = left_time // 3600
        left_m = (left_time - left_h * 3600) // 60
        left_s = left_time - left_h * 3600 - left_m * 60
        s_left_time = '%dh:%dm:%.3fs' % (left_h, left_m, left_s)
    elif left_time > 60:
        left_m = left_time // 60
        left_s = left_time - left_m * 60
        s_left_time = '%dm:%.3fs' % (left_m, left_s)
    else:
        s_left_time = '%.3fs' % (left_time)

    return s_left_time

def eval_split(all_model_cnns, all_models, loader, opt, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    batch_size = eval_kwargs.get('batch_size', 1)
    coco_caption_path = eval_kwargs.get('coco_caption_path', 'coco-caption')
    seq_per_img = 5

    print('start eval ...')

    split = 'test'
    loader.reset_iterator(split)
    n = 0
    predictions = []
    vocab = loader.get_vocab()

    while True:

        start = time.time()

        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        # images = torch.from_numpy(data['images']).cuda()
        # images = utils.prepro_norm(images, False)
        # images = Variable(images, requires_grad=False)

        images = data['images']

        if opt.eval_type == 1: # server
            images = torch.from_numpy(images).cuda()
            images = utils.prepro_norm(images, False)
            images = Variable(images, requires_grad=False)

        output = sample_beam_ensamble(all_model_cnns, all_models, images, eval_kwargs)

        seq = output[0]

        # sents
        sents = utils.decode_sequence(vocab, seq)

        if opt.eval_type == 0: # local
            labels = data['labels']
            captions = utils.decode_sequence(vocab, labels.data)

            if len(output) == 3:

                threshhold = 0.8
                top_word_count = 50

                prob_w = output[2]
                # print(prob_w[0])
                # print(prob_w[1])
                prob_w_0 = prob_w[1]
                np_prob_w = prob_w_0.data.cpu().numpy()
                np_prob_w_keep = np.argsort(np_prob_w)[::-1][:top_word_count]

                vocab['0'] = '<>'
                tokens = []
                for i in range(len(np_prob_w_keep)):
                    tokens.append(vocab[str(np_prob_w_keep[i])])

            print(tokens)

        for k, sent in enumerate(sents):
            image_id = data['infos'][k]['id']
            if opt.eval_type == 1: #server
                image_id = int(image_id.split('_')[2])
            entry = {'image_id': image_id, 'caption': sent}
            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                if opt.eval_type == 0: # local
                    for caption in captions[k*seq_per_img:(k+1)*seq_per_img]:
                        print('groudtruth : %s' % (caption))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            span_time = time.time() - start
            left_time = (ix1 - ix0) * span_time / batch_size
            s_left_time = format_time(left_time)
            print('evaluating validation preformance... %d/%d %.3fs left:%s' % (ix0, ix1, span_time, s_left_time))

        if data['bounds']['wrapped']:
            break

        print('time {:.3f} s'.format(time.time() - start))


    if opt.eval_type == 0: # local
        lang_stats, str_stats = language_eval(predictions, coco_caption_path, opt.input_anno)
    else:
        lang_stats = None
        str_stats = None

    return predictions, lang_stats, str_stats

def load_infos(opt):
    infos = {}
    if opt.start_from_best is not None and len(opt.start_from_best) > 0:
        print("start best from %s" % (opt.start_from_best))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from_best, opt.id + '_infos_best.pkl')) as f:
            infos = cPickle.load(f)
    elif opt.start_from is not None and len(opt.start_from) > 0:
        print("start from %s" % (opt.start_from))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, opt.id + '_infos.pkl')) as f:
            infos = cPickle.load(f)
    return infos

def init_hidden(model, batch_size, fc_feats):
    # fc_feats (batch_size * size)
    inputs = []

    for i in range(model.num_layers*2):
            inputs.append(Variable(torch.FloatTensor(batch_size, model.rnn_size).copy_(fc_feats.data)).cuda())

    return inputs

def get_logprob(logprobs,logprob_pool_type=0):

    if logprob_pool_type == 0:
        logprob = torch.cat([_.unsqueeze(1) for _ in logprobs], 1)
        logprob = logprob.mean(1)
    elif logprob_pool_type == 1:
        logprob = torch.cat([_.unsqueeze(1) for _ in logprobs], 1)
        logprob = logprob.max(1)[0]
    elif logprob_pool_type == 2:
        logprob = logprobs[-1]

    logprob = logprob.view_as(logprobs[-1])

    return logprob


def sample_beam_ensamble(model_cnns, models, images, opt={}):

    beam_size = opt.get('beam_size', 10)
    batch_size = opt.get('batch_size', 2)
    input_type = opt.get('input_type', 0)

    ensamble_num = len(models)
    model_fisrt = models[0]

    seq_length = model_fisrt.seq_length
    input_encoding_size = model_fisrt.input_encoding_size
    att_size = model_fisrt.att_size

    list_fc_feats = []
    list_att_feats = []

    seqs = torch.LongTensor(ensamble_num, batch_size, seq_length).zero_()
    seqLogprobs = torch.FloatTensor(ensamble_num, batch_size, seq_length).zero_()


    for i in range(ensamble_num):

        fc_feats, att_feats = model_cnns[i](images)

        fc_feats_ext, att_feats_ext = models[i].embed_feats(fc_feats, att_feats)

        list_fc_feats.append(fc_feats_ext)
        list_att_feats.append(att_feats_ext)

        # visual_rnn.show_all_probs(att_feats[0].data.cpu().numpy())
        # visual_rnn.show_all_probs(fc_feats_ext.data.cpu().numpy())


    for k in range(batch_size):

        list_beam_seq = []
        list_beam_seq_logprobs = []
        list_beam_logprobs_sum = []
        list_done_beams = []

        list_beam_fc_feats = []
        list_beam_att_feats = []
        list_embed_inputs = []

        list_state = []

        for i in range(ensamble_num):

            model = models[i]

            beam_seq = torch.LongTensor(seq_length, beam_size).cuda().zero_()
            beam_seq_logprobs = torch.FloatTensor(seq_length, beam_size).cuda().zero_()
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []

            beam_fc_feats = list_fc_feats[i][k:k+1].expand(beam_size, input_encoding_size).contiguous()
            beam_att_feats = list_att_feats[i][k:k+1].expand(beam_size, att_size, input_encoding_size).contiguous()
            embed_inputs = torch.LongTensor(beam_size, seq_length + 2).cuda()

            state = init_hidden(model, beam_size, beam_fc_feats)

            list_beam_seq.append(beam_seq)
            list_beam_seq_logprobs.append(beam_seq_logprobs)
            list_beam_logprobs_sum.append(beam_logprobs_sum)
            list_done_beams.append(done_beams)

            list_beam_fc_feats.append(beam_fc_feats)
            list_beam_att_feats.append(beam_att_feats)
            list_embed_inputs.append(embed_inputs)

            list_state.append(state)

        list_logprob = []

        for t in range(seq_length+2):

            temp_logprob = []

            for i in range(ensamble_num):

                model = models[i]

                beam_seq = list_beam_seq[i]
                beam_seq_logprobs = list_beam_seq_logprobs[i]
                beam_logprobs_sum = list_beam_logprobs_sum[i]
                done_beams = list_done_beams[i]

                beam_fc_feats = list_beam_fc_feats[i]
                beam_att_feats = list_beam_att_feats[i]
                embed_inputs = list_embed_inputs[i]

                state = list_state[i]

                if t == 0:
                    # xt = torch.cat((beam_fc_feats, beam_fc_feats), 1)
                    xt = beam_fc_feats
                elif t == 1:
                    it = torch.LongTensor(beam_size).cuda().zero_()
                    if input_type == 0:
                        embed_inputs[:, t] = it
                        it = Variable(it, requires_grad=False)
                        # batch_size * rnn_size
                        concat_temp = model.embed(it)
                        # text-conditional image embedding
                        text_condition = model.embed_tc(it)

                        image_temp = F.softmax(beam_fc_feats * text_condition)

                        # concatenate the textual feature and the guidance
                        # xt = torch.cat((concat_temp, image_temp), 1)
                        xt = concat_temp + image_temp
                        # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
                    elif input_type == 1:
                        xt = model.embed(Variable(it))
                else:
                    logprob = get_logprob(list_logprob[t-1], 1)
                    # logprob = logprobs[-1]
                    logprobsf = logprob.float()
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

                        if v['c'] == 0 or t == seq_length+1:
                            done_beams.append({'seq':beam_seq[:, vix].clone(),
                                               'logps':beam_seq_logprobs[:, vix].clone(),
                                               'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t-2]

                    if input_type == 0:
                        embed_inputs[:, t] = it
                        concat_temp = model.embed(Variable(it))
                        # batch_size * (t-1)
                        prev_inputs = embed_inputs[:, 1:t]
                        # batch_size * (t-1) * input_encoding_size
                        lookup_table_out = model.embed_tc(Variable(prev_inputs))
                        # batch_size * input_encoding_size
                        text_condition = lookup_table_out.mean(1)

                        image_temp = F.softmax(beam_fc_feats * text_condition)
                        # concatenate the textual feature and the guidance
                        # xt = torch.cat((concat_temp, image_temp), 1)
                        xt = concat_temp + image_temp
                        # xt = torch.cat((concat_temp.unsqueeze(1), image_temp.unsqueeze(1)), 1).mean(1).squeeze()
                    elif input_type == 1:
                        xt = model.embed(Variable(it.cuda()))

                        if model.phrase_type == 3:
                            # t - 2 + 3
                            embed_inputs[:, t - 2 + model.context_len - 1] = xt.data
                            embed_inputs_v = Variable(embed_inputs[:, t - 2:t - 2 + model.context_len])
                            # t - 2 : t - 2 + 4
                            phrase_it = model.phraseEmbed(embed_inputs_v)
                            phrase_it1 = model.phraseEmbed1(embed_inputs_v)
                            xt = xt + phrase_it + phrase_it1
                        elif model.phrase_type > 0:
                            embed_inputs[:, t - 2 + model.context_len - 1] = xt.data
                            phrase_it = model.phraseEmbed(Variable(embed_inputs[:, t - 2: t - 2 + model.context_len]))
                            xt += phrase_it
                if t >= 2:
                    state = new_state

                core_result = model.core(xt, beam_att_feats, state)
                list_state[i] = core_result[0]
                logprobs = core_result[1]
                if len(core_result) == 3:
                    prob_w = core_result[2]
                if type(logprobs) == type([]):
                    logprob = get_logprob(logprobs,0)
                else:
                    logprob = logprobs
                temp_logprob.append(logprob)

            list_logprob.append(temp_logprob)

        for i in range(ensamble_num):
            done_beams = list_done_beams[i]
            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seqs[i, k, :] = done_beams[0]['seq']
            seqLogprobs[i, k, :] = done_beams[0]['logps']

    seqLogprob, seq_ind = seqLogprobs.max(0)
    seq_ind = seq_ind.view(1, seq_ind.size(0), seq_ind.size(1))
    seq = seqs.gather(0, seq_ind)

    seq = seq.view(batch_size, seq_length)
    seqLogprob = seqLogprob.view(batch_size, seq_length)

    # seq = seqs[0].view(batch_size, seq_length)
    # seqLogprob = seqLogprobs[0].view(batch_size, seq_length)

    return seq, seqLogprob, prob_w


def main():

    opt = parse_args()

    # make dirs
    print(opt.output_dir)
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    # print(opt)

    all_model_cnns = []
    all_models = []

    for i in range(len(opt.ids)):

        # id
        opt.id = opt.ids[i]

        # Load infos
        infos = load_infos(opt)

        ignore = ["id", "batch_size", "beam_size", "start_from", "start_from_best", "input_json",
                  "input_h5", "input_anno", "images_root", "coco_caption_path"]

        for k in vars(infos['opt']).keys():
            if k not in ignore:
                vars(opt).update({k: vars(infos['opt'])[k]})

        print(opt)

        # Setup the model
        model_cnn = models.setup_cnn(opt)
        # model_cnn.cuda()
        model_cnn = nn.DataParallel(model_cnn.cuda())

        model = models.setup(opt)
        model.cuda()

        # Make sure in the evaluation mode
        model_cnn.eval()
        model.eval()

        all_model_cnns.append(model_cnn)
        all_models.append(model)

    if opt.eval_type == 0: # local test

        print('eval local')

        loader = DataLoaderThreadNew(opt, is_only_test=True)

        # Set sample options
        predictions, lang_stats, str_stats = eval_split(all_model_cnns, all_models, loader, opt, vars(opt))

        save_result(opt.output_dir, str_stats, predictions)


    elif opt.eval_type == 1: # server

        print('eval server')

        for dataset in opt.datasets:

            loader = DataLoaderRaw({'folder_path': os.path.join(opt.image_folder, dataset),
                                    'batch_size': opt.batch_size,
                                    'start': opt.start,
                                    'num': opt.num})
            loader.ix_to_word = infos['vocab']

            # Set sample options
            predictions, lang_stats, str_stats = eval_split(all_model_cnns, all_models, loader, opt, vars(opt))

            path_json = opt.output_dir + '/captions_' + dataset + '_ensemble_' + str(opt.start) + '_results.json'

            json.dump(predictions, open(path_json, 'w'))

if __name__ == '__main__':
    main()