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
from data.DataLoaderThreadBu import *

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


def parse_args():
    # Input arguments and options
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--input_json', type=str, default='/home/amds/caption/dataset/aic/data_aic5.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_h5', type=str, default='/home/amds/caption/dataset/aic/data_aic5.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_anno', type=str, default='/home/amds/caption/dataset/aic/anno_aic5.json',
                        help='')
    parser.add_argument('--images_root', default='/home/amds/caption/dataset/aic',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    # Basic options
    parser.add_argument('--batch_size', type=int, default=2,
                    help='if > 0 then overrule, otherwise load from checkpoint.')

    # Sampling options
    parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # For evaluation on a folder of images:
    parser.add_argument('--aic_caption_path', type=str, default='/home/amds/caption/AI_Challenger/Evaluation/caption_eval/coco_caption',
                    help='aic_caption_path')
    parser.add_argument('--image_folder', type=str, default='/home/amds/caption/dataset/aic/ai_challenger_caption_test1_20170923',
                    help='If this is nonempty then will predict on the images in this folder path'
                         '/home/amds/caption/dataset/aic/ai_challenger_caption_test1_20170923'
                         '')
    parser.add_argument('--start_from_best', type=str, default='/home/amds/data/models/best_model',
                        help="")
    parser.add_argument('--output_dir', type=str, default='/home/amds/data/models/best_model/ensemble_result/',
                        help='')
    parser.add_argument('--output_beam_dir', type=str, default='/home/amds/data/models/best_model/ensemble_result/',
                        help='')
    parser.add_argument('--datasets', type=str, default='caption_test_b_images_20171120',
                        help='caption_test1_images_20170923')
    # misc
    # ['soft_att_weight_1_rl_1', 'dasc_sup_1_rl_2','dasc_sup_1_rl_3','dasc_sup_1_rl_4']
    # 'aic_AVG', 'aic_Bleu_4', 'aic_CIDEr', 'aic_ROUGE_L'
    parser.add_argument('--ids', type=str, default='a1,a2',
                        help='')
    parser.add_argument('--eval_type', type=int, default=1,
                        help='0 local, 1 server')
    parser.add_argument('--input_type', type=int, default=0,
                        help='0 sup, 1 weight')

    parser.add_argument('--start', type=int, default=0,
                        help='')
    parser.add_argument('--num', type=int, default=2,
                        help='')

    parser.add_argument('--ensemble_type', type=int, default=0,
                        help='0 mean, 1 max')

    # bottom up attention
    parser.add_argument('--input_bu', type=str, default='/home/amds/data/code/bottom-up-attention/output/aic_val_resnet101_fasterRcnn_lmdb',
                        help='')
    parser.add_argument('--bu_size', type=int, default=36,
                        help='use bottom up attention')
    parser.add_argument('--bu_feat_size', type=int, default=2048,
                        help='use bottom up attention')
    parser.add_argument('--use_image', type=bool, default=True,
                        help='')
    parser.add_argument('--use_bu_att', type=bool, default=True,
                        help='')

    # beam type
    parser.add_argument('--beam_type', type=int, default=0,
                        help='')

    return parser.parse_args()



def language_eval_aic(id, preds, aic_caption_path, reference_file):

    sys.path.append(aic_caption_path)

    from pycxtools.coco import COCO
    from pycxevalcap.eval import COCOEvalCap

    """Compute m1_score"""
    m1_score = {}
    m1_score['error'] = 0

    coco = COCO(reference_file)

    tmp_name = "temp_" + id
    json_predictions_file = tmp_name + '.json'
    json.dump(preds, open(json_predictions_file, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    coco_res = coco.loadRes(json_predictions_file)

    # create coco_eval object.
    coco_eval = COCOEvalCap(coco, coco_res)

    coco_eval.params['image_id'] = coco_res.getImgIds()

    # evaluate results
    coco_eval.evaluate()

    # delete the temp file
    os.system('rm ' + json_predictions_file)

    # print output evaluation scores
    # for metric, score in coco_eval.eval.items():
    #     print('%s: %.3f'%(metric, score))
    #     m1_score[metric] = score

    # create output dictionary
    out = {}
    out_str = []

    metrics = ["CIDEr", "ROUGE_L", "METEOR", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    # metrics = ["CIDEr", "ROUGE_L", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    for metric in metrics:
        score = coco_eval.eval[metric]
        out[metric] = score
        out_str.append(str(score))
    str_result = ','.join(out_str)

    return out, str_result

def save_beam_vis_result(output_beam_dir, filename, beam_vis):
    beam_vis_file = os.path.join(output_beam_dir, filename)
    with open(beam_vis_file, 'w') as f:
        json.dump(beam_vis, f)

def save_result(output_dir, str_stats, predictions):
    eval_result_file = os.path.join(output_dir, "eval_ensemble.csv")
    with open(eval_result_file, 'a') as f:
        f.write(str_stats + "\n")

    predictions_file = os.path.join(output_dir, "eval_ensemble.json")
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
    aic_caption_path = eval_kwargs.get('aic_caption_path', 'coco-caption')
    num = eval_kwargs.get('num', 5000)
    eval_type = eval_kwargs.get('eval_type', 0)


    print('start eval ...')

    split = 'val'
    loader.reset_iterator(split)
    n = 0
    predictions = []
    vocab = loader.get_vocab()

    beam_vis = {}
    total_predicted_ids = []
    total_beam_parent_ids = []
    total_scores = []
    total_ids = []
    total_sents = []

    while True:

        start = time.time()

        start_data = time.time()
        data = loader.get_batch(split, batch_size)
        print('data time {:.3f} s'.format(time.time() - start_data))

        n = n + batch_size

        # images = torch.from_numpy(data['images']).cuda()
        # images = utils.prepro_norm(images, False)
        # images = Variable(images, requires_grad=False)

        images = data['images']

        if opt.eval_type == 1: # server
            images = torch.from_numpy(images).cuda()
            images = utils.prepro_norm(images, False)
            images = Variable(images, requires_grad=False)

        if models.has_bu(opt.caption_model):
            if opt.eval_type == 0: # local
                bu_feats = data['bus']
            elif opt.eval_type == 1: # server
                bus = torch.from_numpy(data['bus']).cuda().float()
                bu_feats = Variable(bus, requires_grad=False)
            seqs, _, batch_predicted_ids, batch_beam_parent_ids, batch_scores = sample_beam_ensamble_with_bu(all_model_cnns, all_models, images, bu_feats, eval_kwargs)
        else:
            seqs, _, batch_predicted_ids, batch_beam_parent_ids, batch_scores = sample_beam_ensamble(all_model_cnns, all_models, images, eval_kwargs)

        # sents
        sents = utils.decode_sequence_aic(vocab, seqs)

        for k, sent in enumerate(sents):
            print(data['infos'][k])
            if opt.eval_type == 0: # local
                image_id = data['infos'][k]['image_id']
            elif opt.eval_type == 1: # server
                image_id = data['infos'][k]['id']
            #if opt.eval_type == 1: # server
            #    image_id = int(image_id.split('_')[2])
            entry = {'image_id': image_id, 'caption': sent}

            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

            total_predicted_ids.append(batch_predicted_ids[k])
            total_beam_parent_ids.append(batch_beam_parent_ids[k])
            total_scores.append(batch_scores[k])
            total_ids.append(image_id)
            total_sents.append(sent)


        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num != -1:
            ix1 = min(ix1, num)

        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            span_time = time.time() - start
            left_time = (ix1 - ix0) * span_time / batch_size
            s_left_time = format_time(left_time)
            print('evaluating validation preformance... %d/%d %.3fs left:%s' % (ix0, ix1, span_time, s_left_time))


        if data['bounds']['wrapped']:
            break
        if n != -1 and n >= num:
            break

        print('time {:.3f} s'.format(time.time() - start))

    beam_vis["predicted_ids"] = total_predicted_ids
    beam_vis["beam_parent_ids"] = total_beam_parent_ids
    beam_vis["scores"] = total_scores
    beam_vis["ids"] = total_ids
    beam_vis["vocab"] = vocab
    beam_vis["sents"] = total_sents

    # print(beam_vis)

    if opt.eval_type == 0: # local
        lang_stats, str_stats = language_eval_aic("ensemble", predictions, aic_caption_path, opt.input_anno)
    else:
        lang_stats = None
        str_stats = None

    return predictions, lang_stats, str_stats, beam_vis

def load_infos(opt):
    infos = {}
    if opt.start_from_best is not None and len(opt.start_from_best) > 0:
        print("start best from %s" % (opt.start_from_best))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from_best, opt.id)) as f:
            infos = cPickle.load(f)
    elif opt.start_from is not None and len(opt.start_from) > 0:
        print("start from %s" % (opt.start_from))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, opt.id)) as f:
            infos = cPickle.load(f)
    return infos

def init_hidden(model, batch_size, fc_feats):
    # fc_feats (batch_size * size)
    inputs = []

    for i in range(model.num_layers*2):
            inputs.append(Variable(torch.FloatTensor(batch_size, model.rnn_size).copy_(fc_feats.data)).cuda())

    return inputs

# 0 mean
# 1 max
# 2 weights
def get_logprob(logprobs, logprob_pool_type=0):

    weights = []

    if logprob_pool_type == 0:
        logprob = torch.cat([_.unsqueeze(1) for _ in logprobs], 1)
        logprob = logprob.mean(1)
    elif logprob_pool_type == 1:
        logprob = torch.cat([_.unsqueeze(1) for _ in logprobs], 1)
        logprob = logprob.max(1)[0]
    elif logprob_pool_type == 2:
        logprob = logprobs[-1]
    elif logprob_pool_type == 3:
        logprob = torch.cat([logprobs[i].unsqueeze(1)*weights[i] for i in range(len(logprobs))], 1)
        logprob = logprob.max(1)[0]

    logprob = logprob.view_as(logprobs[-1])

    return logprob


def sample_beam_ensamble(model_cnns, models, images, opt={}):

    beam_size = opt.get('beam_size', 10)
    batch_size = opt.get('batch_size', 2)
    input_type = opt.get('input_type', 0)
    ensemble_type = opt.get('ensemble_type', 0)

    print('beam_size', beam_size, 'batch_size', batch_size)

    ensamble_num = len(models)
    model_fisrt = models[0]

    seq_length = model_fisrt.seq_length
    input_encoding_size = model_fisrt.input_encoding_size
    att_size = model_fisrt.att_size

    list_fc_feats = []
    list_att_feats = []

    seqs = torch.LongTensor(ensamble_num, batch_size, seq_length).zero_()
    seqLogprobs = torch.FloatTensor(ensamble_num, batch_size, seq_length).zero_()

    start_cnn = time.time()

    for i in range(ensamble_num):

        fc_feats, att_feats = model_cnns[i](images)

        fc_feats_ext, att_feats_ext = models[i].embed_feats(fc_feats, att_feats)

        list_fc_feats.append(fc_feats_ext)
        list_att_feats.append(att_feats_ext)

        # visual_rnn.show_all_probs(att_feats[0].data.cpu().numpy())
        # visual_rnn.show_all_probs(fc_feats_ext.data.cpu().numpy())

    print('cnn time {:.3f} s'.format(time.time() - start_cnn))

    batch_predicted_ids = []
    batch_beam_parent_ids = []
    batch_scores = []

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

        predicted_ids = []
        beam_parent_ids = []
        scores = []

        for t in range(seq_length+2):

            temp_logprob = []

            pred_ids = []
            parent_ids = []
            scor = []

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
                    # batch_size * (vocab_size + 1)
                    logprob = get_logprob(list_logprob[t-1], ensemble_type)
                    # batch_size * (vocab_size + 1)
                    # logprob = logprobs[-1]
                    logprobsf = logprob.float()
                    # ys : beam_size * (vocab_size + 1)
                    # ix : beam_size * (vocab_size + 1)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    # min(beam_size, (vocab_size + 1))
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 2:
                        rows = 1
                    # cols : (vocab_size + 1)
                    for c in range(cols):
                        # rows : beam_size
                        for q in range(rows):
                            # q : beam index
                            # local_logprob : current word prob
                            local_logprob = ys[q, c]
                            # beam_logprobs_sum :
                            # local_logprob : prob
                            # candidate_logprob : beam total logprob
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            # c : word index
                            # q : beam index
                            # p : total logprob
                            # r : current logprob
                            candidates.append({'c':ix.data[q,c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    # sorted by the total logprob
                    # candidates : 2 * (vocab_size + 1)
                    candidates = sorted(candidates, key=lambda x:-x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]

                    if t > 2:
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()

                    # select only two top logprob
                    for vix in range(beam_size):
                        # sorted candidates
                        v = candidates[vix]

                        # history
                        pred_ids.append(v['c'].data)
                        scor.append(v['r'].data)

                        if t > 2:
                            beam_seq[:t-2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]
                            parent_ids.append(beam_seq_prev[:, v['q']].data)
                        else:
                            parent_ids.append(0)

                        for state_ix in range(len(new_state)):
                            new_state[state_ix][vix] = state[state_ix][v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t-2, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-2, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        # sentence end
                        if v['c'] == 0 or t == seq_length + 1:
                            done_beams.append({'seq' : beam_seq[:, vix].clone(),
                                               'logps' : beam_seq_logprobs[:, vix].clone(),
                                               'p' : beam_logprobs_sum[vix]})

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
                if type(logprobs) == type([]):
                    logprob = get_logprob(logprobs,0)
                else:
                    logprob = logprobs
                temp_logprob.append(logprob)

            list_logprob.append(temp_logprob)

            predicted_ids.append(pred_ids)
            beam_parent_ids.append(parent_ids)
            scores.append(scor)

        for i in range(ensamble_num):
            # done_beams :
            done_beams = list_done_beams[i]
            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seqs[i, k, :] = done_beams[0]['seq']
            seqLogprobs[i, k, :] = done_beams[0]['logps']

        batch_predicted_ids.append(predicted_ids)
        batch_beam_parent_ids.append(beam_parent_ids)
        batch_scores.append(scores)

    seqLogprob, seq_ind = seqLogprobs.max(0)
    seq_ind = seq_ind.view(1, seq_ind.size(0), seq_ind.size(1))
    seq = seqs.gather(0, seq_ind)

    seq = seq.view(batch_size, seq_length)
    seqLogprob = seqLogprob.view(batch_size, seq_length)

    # seq = seqs[0].view(batch_size, seq_length)
    # seqLogprob = seqLogprobs[0].view(batch_size, seq_length)

    return seq, seqLogprob, batch_predicted_ids, batch_beam_parent_ids, batch_scores



def sample_beam_ensamble_with_bu(model_cnns, models, images, bu_feats, opt={}):

    beam_size = opt.get('beam_size', 10)
    batch_size = opt.get('batch_size', 2)
    input_type = opt.get('input_type', 0)
    bu_size = opt.get('bu_size', 36)
    beam_type = opt.get('beam_type', 0)

    print('beam_size', beam_size, 'batch_size', batch_size)

    ensamble_num = len(models)
    model_fisrt = models[0]

    seq_length = model_fisrt.seq_length
    input_encoding_size = model_fisrt.input_encoding_size
    att_size = model_fisrt.att_size

    list_fc_feats = []
    list_att_feats = []
    list_bu_feats = []

    seqs = torch.LongTensor(ensamble_num, batch_size, seq_length).zero_()
    seqLogprobs = torch.FloatTensor(ensamble_num, batch_size, seq_length).zero_()

    start_cnn = time.time()

    for i in range(ensamble_num):

        fc_feats, att_feats = model_cnns[i](images)

        fc_feats_ext, att_feats_ext, bu_feats_ext = models[i].embed_feats(fc_feats, att_feats, bu_feats)

        list_fc_feats.append(fc_feats_ext)
        list_att_feats.append(att_feats_ext)
        list_bu_feats.append(bu_feats_ext)

        # visual_rnn.show_all_probs(att_feats[0].data.cpu().numpy())
        # visual_rnn.show_all_probs(fc_feats_ext.data.cpu().numpy())

    print('cnn time {:.3f} s'.format(time.time() - start_cnn))

    batch_predicted_ids = []
    batch_beam_parent_ids = []
    batch_scores = []

    for k in range(batch_size):

        list_beam_seq = []
        list_beam_seq_logprobs = []
        list_beam_logprobs_sum = []
        list_done_beams = []

        list_beam_fc_feats = []
        list_beam_att_feats = []
        list_beam_bu_feats = []
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
            beam_bu_feats = list_bu_feats[i][k:k+1].expand(beam_size, bu_size, input_encoding_size).contiguous()
            embed_inputs = torch.LongTensor(beam_size, seq_length + 2).cuda()

            state = init_hidden(model, beam_size, beam_fc_feats)

            list_beam_seq.append(beam_seq)
            list_beam_seq_logprobs.append(beam_seq_logprobs)
            list_beam_logprobs_sum.append(beam_logprobs_sum)
            list_done_beams.append(done_beams)

            list_beam_fc_feats.append(beam_fc_feats)
            list_beam_att_feats.append(beam_att_feats)
            list_beam_bu_feats.append(beam_bu_feats)
            list_embed_inputs.append(embed_inputs)

            list_state.append(state)

        list_logprob = []

        predicted_ids = []
        beam_parent_ids = []
        scores = []

        for t in range(seq_length+2):

            # ensemble
            temp_logprob = []

            pred_ids = []
            parent_ids = []
            scor = []

            for i in range(ensamble_num):

                model = models[i]

                beam_seq = list_beam_seq[i]
                beam_seq_logprobs = list_beam_seq_logprobs[i]
                beam_logprobs_sum = list_beam_logprobs_sum[i]
                done_beams = list_done_beams[i]

                beam_fc_feats = list_beam_fc_feats[i]
                beam_att_feats = list_beam_att_feats[i]
                beam_bu_feats = list_beam_bu_feats[i]
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
                    logprob = get_logprob(list_logprob[t-1],1)
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

                        # history
                        pred_ids.append(v['c'])
                        scor.append(v['r'])

                        if t > 2:
                            beam_seq[:t-2,vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]
                            parent_ids.append(v['q'])
                        else:
                            parent_ids.append(0)

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

                            if beam_type == 1:
                                beam_logprobs_sum[vix] = -100


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

                core_result = model.core(xt, beam_att_feats, beam_bu_feats, state)
                list_state[i] = core_result[0]
                logprobs = core_result[1]
                if type(logprobs) == type([]):
                    logprob = get_logprob(logprobs,0)
                else:
                    logprob = logprobs
                # logprob : beam_size * (vocab_size + 1)
                # temp_logprob : ensemble_size
                temp_logprob.append(logprob)

            # list_logprob : (seq_length + 2) * ensemble_size * beam_size * (vocab_size + 1)
            list_logprob.append(temp_logprob)

            if t >= 2:
                predicted_ids.append(pred_ids)
                beam_parent_ids.append(parent_ids)
                scores.append(scor)

        for i in range(ensamble_num):
            done_beams = list_done_beams[i]
            done_beams = sorted(done_beams, key=lambda x: -x['p'])
            seqs[i, k, :] = done_beams[0]['seq']
            seqLogprobs[i, k, :] = done_beams[0]['logps']

        batch_predicted_ids.append(predicted_ids)
        batch_beam_parent_ids.append(beam_parent_ids)
        batch_scores.append(scores)

    seqLogprob, seq_ind = seqLogprobs.max(0)
    if seq_ind.size(0) > 1:
        seq_ind = seq_ind.view(1, seq_ind.size(0), seq_ind.size(1))
    seq = seqs.gather(0, seq_ind)

    seq = seq.view(batch_size, seq_length)
    seqLogprob = seqLogprob.view(batch_size, seq_length)

    # seq = seqs[0].view(batch_size, seq_length)
    # seqLogprob = seqLogprobs[0].view(batch_size, seq_length)

    return seq, seqLogprob, batch_predicted_ids, batch_beam_parent_ids, batch_scores


def main():

    opt = parse_args()

    opt.datasets = opt.datasets.split(',')
    opt.ids = opt.ids.split(',')

    # make dirs
    print(opt.output_dir)
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    print(opt.output_beam_dir)
    if not os.path.isdir(opt.output_beam_dir):
        os.makedirs(opt.output_beam_dir)

    # print(opt)

    all_model_cnns = []
    all_models = []

    for i in range(len(opt.ids)):

        # id
        opt.id = opt.ids[i]

        # Load infos
        infos = load_infos(opt)

        ignore = ["id", "batch_size", "beam_size", "start_from_best", "input_json",
                  "input_h5", "input_anno", "images_root", "aic_caption_path", "input_bu"]

        for k in vars(infos['opt']).keys():
            if k not in ignore:
                vars(opt).update({k: vars(infos['opt'])[k]})

        opt.relu_type = 0

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

        if models.has_bu(opt.caption_model):
            loader = DataLoaderThreadBu(opt)
        else:
            loader = DataLoaderThreadNew(opt)

        # Set sample options
        predictions, lang_stats, str_stats, beam_vis = eval_split(all_model_cnns, all_models, loader, opt, vars(opt))

        save_result(opt.output_dir, str_stats, predictions)

        save_beam_vis_result(opt.output_beam_dir, "eval_beam_vis.json", beam_vis)


    elif opt.eval_type == 1: # server

        print('eval server')

        for dataset in opt.datasets:

            print(os.path.join(opt.image_folder, dataset))

            loader = DataLoaderRaw({'folder_path': os.path.join(opt.image_folder, dataset),
                                    'batch_size': opt.batch_size,
                                    'start': opt.start,
                                    'num': opt.num,
                                    'use_bu_att': opt.use_bu_att,
                                    'input_bu': opt.input_bu,
                                    'bu_size': opt.bu_size,
                                    'bu_feat_size': opt.bu_feat_size})

            loader.ix_to_word = infos['vocab']

            # Set sample options
            predictions, lang_stats, str_stats, beam_vis = eval_split(all_model_cnns, all_models, loader, opt, vars(opt))

            path_json = opt.output_dir + '/captions_' + dataset + str(opt.start) + '_ensemble_results.json'

            json.dump(predictions, open(path_json, 'w'))

            save_beam_vis_result(opt.output_beam_dir, dataset + str(opt.start) + "_beam_size_" + str(opt.beam_size) + "_beam_type_" + str(opt.beam_type) + "_eval_beam_vis.json", beam_vis)


if __name__ == '__main__':
    main()