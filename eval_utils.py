from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import misc.utils as utils
import sys
import math
import gc
import models

reload(sys)
sys.setdefaultencoding('utf8')

def language_eval(id, preds, coco_caption_path, input_anno):

    sys.path.append(coco_caption_path)
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # annFile = os.path.join(coco_caption_path, 'annotations/captions_val2014.json')

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    random.seed(time.time())
    # tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    tmp_name = "temp_" + id
    print(tmp_name)
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

    #     metrics = ["CIDEr", "ROUGE_L", "METEOR", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    metrics = ["CIDEr", "ROUGE_L", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    for metric in metrics:
        score = cocoEval.eval[metric]
        out[metric] = score
        out_str.append(str(score))
    str_result = ','.join(out_str)

    return out, str_result

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

def compute_loss(crit, model, caption_model, seq_per_img, fc_expander, att_expander, bu_expander, fc_feats, att_feats, bu_feats, labels, masks, tokens):

    if models.is_only_fc_feat(caption_model):
        if seq_per_img > 1:
            fc_feats_ext = fc_expander(fc_feats)
        else:
            fc_feats_ext = fc_feats
        batch_outputs = model(fc_feats_ext, labels)
    elif models.is_only_att_feat(caption_model):
        if seq_per_img > 1:
            att_feats_ext = att_expander(att_feats)
        else:
            att_feats_ext = att_feats
        batch_outputs = model(att_feats_ext, labels)
    elif caption_model == "SCST":
        if seq_per_img > 1:
            fc_feats_ext = fc_expander(fc_feats)
            att_feats_ext = att_expander(att_feats)
        else:
            fc_feats_ext = fc_feats
            att_feats_ext = att_feats
        batch_outputs, _ = model(fc_feats_ext, att_feats_ext, labels, "train")
    elif models.is_prob_weight(caption_model):
        if models.has_sub_region_bu(caption_model):
            if seq_per_img > 1:
                fc_feats_ext = fc_expander(fc_feats)
                att_feats_ext = att_expander(att_feats)
                bu_feats_ext = bu_expander(bu_feats)
            else:
                fc_feats_ext = fc_feats
                att_feats_ext = att_feats
                bu_feats_ext = bu_feats

            batch_outputs, prob_w = model(fc_feats_ext, att_feats_ext, bu_feats_ext, labels)
        else:
            if seq_per_img > 1:
                fc_feats_ext = fc_expander(fc_feats)
                att_feats_ext = att_expander(att_feats)
            else:
                fc_feats_ext = fc_feats
                att_feats_ext = att_feats

            if models.has_bu(caption_model):
                if seq_per_img > 1:
                    bu_feats_ext = bu_expander(bu_feats)
                else:
                    bu_feats_ext = bu_feats
                batch_outputs, prob_w = model(fc_feats_ext, att_feats_ext, bu_feats_ext, labels)
            else:
                batch_outputs, prob_w = model(fc_feats_ext, att_feats_ext, labels)
    elif models.is_prob_weight_mul_out(caption_model):
        if seq_per_img > 1:
            fc_feats_ext = fc_expander(fc_feats)
            att_feats_ext = att_expander(att_feats)
        else:
            fc_feats_ext = fc_feats
            att_feats_ext = att_feats

        if models.has_bu(caption_model):
            if seq_per_img > 1:
                bu_feats_ext = bu_expander(bu_feats)
            else:
                bu_feats_ext = bu_feats
            batch_outputs, prob_w = model(fc_feats_ext, att_feats_ext, bu_feats_ext, labels)
        else:
            batch_outputs, prob_w = model(fc_feats_ext, att_feats_ext, labels)
    else:
        if seq_per_img > 1:
            fc_feats_ext = fc_expander(fc_feats)
            att_feats_ext = att_expander(att_feats)
        else:
            fc_feats_ext = fc_feats
            att_feats_ext = att_feats

        if models.has_bu(caption_model):
            if seq_per_img > 1:
                bu_feats_ext = bu_expander(bu_feats)
            else:
                bu_feats_ext = bu_feats
            batch_outputs = model(fc_feats_ext, att_feats_ext, bu_feats_ext, labels)
        else:
            batch_outputs = model(fc_feats_ext, att_feats_ext, labels)

    if models.is_prob_weight(caption_model) or models.is_prob_weight_mul_out(caption_model):
        loss = crit(batch_outputs, labels, masks, prob_w, tokens)
    else:
        loss = crit(batch_outputs, labels, masks)
    loss.backward()

    return loss.data[0]

def compute_cnn_feats(caption_model, model_cnn, images):

    fc_feats = None
    att_feats = None
    bu_feats = None

    if models.is_only_fc_feat(caption_model):
        fc_feats = model_cnn(images)
    elif models.is_only_att_feat(caption_model):
        att_feats = model_cnn(images)
    elif caption_model == "SCST":
        fc_feats, att_feats = model_cnn(images)
    elif models.is_prob_weight(caption_model):
        if models.has_sub_region_bu(caption_model):
            fc_feats, att_feats, bu_feats = model_cnn(images)
        else:
            fc_feats, att_feats = model_cnn(images)
    elif models.is_prob_weight_mul_out(caption_model):
        fc_feats, att_feats = model_cnn(images)
    else:
        fc_feats, att_feats = model_cnn(images)

    return fc_feats, att_feats, bu_feats

def compute_output(caption_model, beam_size, model, fc_feats, att_feats, bu_feats):
    if models.is_only_fc_feat(caption_model):
        output = model.sample(fc_feats, {'beam_size': beam_size})
    elif models.is_only_att_feat(caption_model):
        output = model.sample(att_feats, {'beam_size': beam_size})
    elif models.has_bu(caption_model) or models.has_sub_region_bu(caption_model) or models.is_prob_weight_mul_out(
            caption_model):
        output = model.sample(fc_feats, att_feats, bu_feats, {'beam_size': beam_size})
    else:
        output = model.sample(fc_feats, att_feats, {'beam_size': beam_size})
    return output


def eval_split(model_cnn, model, crit, loader, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    beam_size = eval_kwargs.get('beam_size', 1)
    coco_caption_path = eval_kwargs.get('coco_caption_path', 'coco-caption')
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 2)
    seq_per_img = eval_kwargs.get('seq_per_img', 5)
    id = eval_kwargs.get('id', '')
    input_anno = eval_kwargs.get('input_anno', '')
    is_compute_val_loss = eval_kwargs.get('is_compute_val_loss', 0)

    # aic caption path
    is_aic_data = eval_kwargs.get('is_aic_data', False)
    aic_caption_path = eval_kwargs.get('aic_caption_path', 'aic-caption')

    # Make sure in the evaluation mode
    model_cnn.eval()
    model.eval()

    if crit is None:
        is_compute_val_loss = 0

    if is_compute_val_loss == 1 and seq_per_img > 1:
        fc_expander = utils.FeatExpander(seq_per_img)
        att_expander = utils.FeatExpander(seq_per_img)
        bu_expander = None
        if models.has_bu(caption_model) or models.has_sub_region_bu(caption_model):
            bu_expander = utils.FeatExpander(seq_per_img)

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    vocab = loader.get_vocab()
    vocab_size = loader.get_vocab_size()
    while True:

        start = time.time()

        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        images = data['images']
        labels = data['labels']
        masks = data['masks']
        tokens = data['tokens']

        images.volatile = True
        labels.volatile = True
        masks.volatile = True
        tokens.volatile = True

        fc_feats, att_feats, bu_feats = compute_cnn_feats(caption_model, model_cnn, images)

        if models.has_bu(caption_model):
            bu_feats = data['bus']
            bu_feats.volatile = True

        if is_compute_val_loss == 1:
            loss = compute_loss(crit, model, caption_model, seq_per_img,
                         fc_expander, att_expander, bu_expander,
                         fc_feats, att_feats, bu_feats,
                         labels, masks, tokens)
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1
        else:
            loss = 0

        output = compute_output(caption_model, beam_size, model, fc_feats, att_feats, bu_feats)

        seq = output[0]

        #
        if type(seq) == type([]):
            seq = seq[-1]
        if is_aic_data:
            sents = utils.decode_sequence_aic(vocab, seq)
            # captions = utils.decode_sequence(vocab, seq)
        else:
            sents = utils.decode_sequence(vocab, seq)
            # captions = utils.decode_sequence(vocab, seq)

        # print(sents)
        # print(captions)

        for k, sent in enumerate(sents):
            if is_aic_data:
                image_id = data['infos'][k]['image_id']
            else:
                image_id = data['infos'][k]['id']
            entry = {'image_id': image_id, 'caption': sent}
            # caption = {'image_id': image_id, 'caption': captions[k]}
            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            span_time = time.time() - start
            left_time = (ix1 - ix0) * span_time / batch_size
            s_left_time = utils.format_time(left_time)
            print('evaluating validation preformance... %d/%d %.3fs left:%s' % (ix0 - 1, ix1, span_time, s_left_time))

        if data['bounds']['wrapped']:
            break
        if n >= val_images_use:
            break


    if lang_eval == 1:
        if is_aic_data:
            lang_stats, str_stats = language_eval_aic(id, predictions, aic_caption_path, input_anno)
        else:
            lang_stats, str_stats = language_eval(id, predictions, coco_caption_path, input_anno)

    # Switch back to training mode
    model_cnn.train()
    model.train()

    if is_compute_val_loss == 1:
        final_loss = loss_sum / loss_evals
    else:
        final_loss = 0

    return final_loss, predictions, lang_stats, str_stats


def eval_split_with_region_bu(model_cnn, model, crit, loader, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    beam_size = eval_kwargs.get('beam_size', 1)
    coco_caption_path = eval_kwargs.get('coco_caption_path', 'coco-caption')
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 2)
    seq_per_img = eval_kwargs.get('seq_per_img', 5)
    id = eval_kwargs.get('id', '')
    input_anno = eval_kwargs.get('input_anno', '')

    # aic caption path
    is_aic_data = eval_kwargs.get('is_aic_data', False)
    aic_caption_path = eval_kwargs.get('aic_caption_path', 'aic-caption')

    # Make sure in the evaluation mode
    model_cnn.eval()
    model.eval()

    if seq_per_img > 1:
        fc_expander = utils.FeatExpander(seq_per_img)
        att_expander = utils.FeatExpander(seq_per_img)
        bu_expander = utils.FeatExpander(seq_per_img)

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    vocab = loader.get_vocab()
    vocab_size = loader.get_vocab_size()
    while True:
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        images = data['images']
        labels = data['labels']
        masks = data['masks']
        tokens = data['tokens']

        fc_feats, att_feats, bu_feats = model_cnn(images)
        if seq_per_img > 1:
            fc_feats_ext = fc_expander(fc_feats)
            att_feats_ext = att_expander(att_feats)
            bu_feats_ext = bu_expander(bu_feats)
        else:
            fc_feats_ext = fc_feats
            att_feats_ext = att_feats
            bu_feats_ext = bu_feats

        batch_outputs, prob_w = model(fc_feats_ext, att_feats_ext, bu_feats_ext, labels)

        loss = crit(batch_outputs, labels, masks, prob_w, tokens)

        loss.backward()

        loss_sum = loss_sum + loss.data[0]
        loss_evals = loss_evals + 1

        seq, _ = model.sample(fc_feats, att_feats, bu_feats, {'beam_size': beam_size})

        #
        if is_aic_data:
            if type(seq) == type([]):
                seq = seq[-1]
            sents = utils.decode_sequence_aic(vocab, seq)
        else:
            sents = utils.decode_sequence(vocab, seq)

        for k, sent in enumerate(sents):
            if is_aic_data:
                image_id = data['infos'][k]['image_id']
            else:
                image_id = data['infos'][k]['id']
            entry = {'image_id': image_id, 'caption': sent}
            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss.data[0]))

        if data['bounds']['wrapped']:
            break
        if n >= val_images_use:
            break


    if lang_eval == 1:
        if is_aic_data:
            lang_stats, str_stats = language_eval_aic(id, predictions, aic_caption_path, input_anno)
        else:
            lang_stats, str_stats = language_eval(id, predictions, coco_caption_path, input_anno)

    # Switch back to training mode
    model_cnn.train()
    model.train()

    return loss_sum / loss_evals, predictions, lang_stats, str_stats


def eval_split_without_cnn(model, crit, loader, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    beam_size = eval_kwargs.get('beam_size', 1)
    coco_caption_path = eval_kwargs.get('coco_caption_path', 'coco-caption')
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 2)
    seq_per_img = eval_kwargs.get('seq_per_img', 5)
    id = eval_kwargs.get('id', '')
    input_anno = eval_kwargs.get('input_anno', '')

    # aic caption path
    is_aic_data = eval_kwargs.get('is_aic_data', False)
    aic_caption_path = eval_kwargs.get('aic_caption_path', 'aic-caption')

    # Make sure in the evaluation mode
    model.eval()

    if seq_per_img > 1:
        bu_expander = utils.FeatExpander(seq_per_img)

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    vocab = loader.get_vocab()
    vocab_size = loader.get_vocab_size()
    while True:
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        labels = data['labels']
        masks = data['masks']
        tokens = data['tokens']
        bu_feats = data['bus']

        if seq_per_img > 1:
            bu_feats_ext = bu_expander(bu_feats)
        else:
            bu_feats_ext = bu_feats

        batch_outputs = model(bu_feats_ext, labels)

        loss = crit(batch_outputs, labels, masks)
        loss.backward()

        loss_sum = loss_sum + loss.data[0]
        loss_evals = loss_evals + 1

        seq, _ = model.sample(bu_feats, {'beam_size': beam_size})

        #
        if is_aic_data:
            sents = utils.decode_sequence_aic(vocab, seq)
        else:
            sents = utils.decode_sequence(vocab, seq)

        for k, sent in enumerate(sents):
            if is_aic_data:
                image_id = data['infos'][k]['image_id']
            else:
                image_id = data['infos'][k]['id']
            entry = {'image_id': image_id, 'caption': sent}
            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss.data[0]))

        if data['bounds']['wrapped']:
            break
        if n >= val_images_use:
            break


    if lang_eval == 1:
        if is_aic_data:
            lang_stats, str_stats = language_eval_aic(id, predictions, aic_caption_path, input_anno)
        else:
            lang_stats, str_stats = language_eval(id, predictions, coco_caption_path, input_anno)

    # Switch back to training mode
    model.train()

    return loss_sum / loss_evals, predictions, lang_stats, str_stats


def eval_split_only(model_cnn, model, crit, loader, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    coco_caption_path = eval_kwargs.get('coco_caption_path', 'coco-caption')
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 2)
    seq_per_img = eval_kwargs.get('seq_per_img', 5)

    # Make sure in the evaluation mode
    model_cnn.eval()
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    vocab = loader.get_vocab()
    vocab_size = loader.get_vocab_size()
    while True:
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        images = data['images']

        if models.is_only_fc_feat(caption_model):
            fc_feats = model_cnn(images)
        elif models.is_only_att_feat(caption_model):
            att_feats = model_cnn(images)
        elif caption_model == "SCST":
            fc_feats, att_feats = model_cnn(images)
        else:
            fc_feats, att_feats = model_cnn(images)

        if models.is_only_fc_feat(caption_model):
            seq, _ = model.sample(fc_feats, {'beam_size': beam_size})
        elif models.is_only_att_feat(caption_model):
            seq, _ = model.sample(att_feats, {'beam_size': beam_size})
        else:
            seq, _ = model.sample(fc_feats, att_feats, {'beam_size': beam_size})

        #
        sents = utils.decode_sequence(vocab, seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            print('evaluating validation preformance... %d/%d' % (ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break
        if n >= val_images_use:
            break

    if lang_eval == 1:
        lang_stats, str_stats = language_eval(dataset, predictions, coco_caption_path)

    # Switch back to training mode
    model_cnn.train()
    model.train()

    return 0, predictions, lang_stats, str_stats
