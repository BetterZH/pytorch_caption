# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import torch
import torch.nn as nn
import numpy as np
# from reward.cider.cider import Cider
from reward.cider.cider import Cider
from reward.bleu.bleu import Bleu
from reward.tokenizer.ptbtokenizer import PTBTokenizer
from collections import OrderedDict
import jieba

CiderD_scorer = None
Bleu_scorer = None
Rouge_scorer = None
Meteor_scorer = None

class FeatExpander(nn.Module):
    def __init__(self, count):
        super(FeatExpander, self).__init__()
        self.count = count

    # batch_size * feat_size
    # batch_size * count * feat_size
    def forward(self, input):
        return input.unsqueeze(1).expand(*((input.size(0), self.count,) + input.size()[1:])).contiguous().view(*((input.size(0) * self.count,) + input.size()[1:]))


def decode_sequence(ix_to_word, seq):

    # batch_size * seq_length * (vocab_size + 1)
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def decode_sequence_aic(ix_to_word, seq):

    # batch_size * seq_length * (vocab_size + 1)
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                # if j >= 1:
                #     txt = txt + ''
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def decode_sequence_aic_cut(ix_to_word, seq):

    # batch_size * seq_length * (vocab_size + 1)
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                # if j >= 1:
                #     txt = txt + ''
                txt = txt + ix_to_word[str(ix)]
            else:
                break

        w = jieba.cut(txt.strip().replace('。', ''), cut_all=False)
        p = ' '.join(w)
        out.append(p)
    return out

def decode_sequence1(ix_to_word, seq):

    D = len(seq)
    txt = ''
    for j in range(D):
        ix = seq[j]
        if ix > 0:
            if j >= 1:
                txt = txt + ' '
            txt = txt + ix_to_word[str(ix)]
        else:
            break
    return txt

def decode_sequence2(ix_to_word, seq):

    D, = seq.size()
    txt = ''
    for j in range(D):
        ix = seq[j]
        if ix > 0:
            if j >= 1:
                txt = txt + ' '
            txt = txt + ix_to_word[str(ix)]
        else:
            break
    return txt

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        group_params = group['params']
        for ind in xrange(len(group_params)):
            param = group_params[ind]
            if param.grad is None:
                pass
                # print('clip_error', param.size())
            else:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def array_to_str(arr):
    out = ''
    has_end = False
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            out += "."
            has_end = True
            break
    if not has_end:
        out += "."
    return out.strip()

def array_to_str_aic(arr, vocab):
    out = ''
    has_end = False
    for i in range(len(arr)):
        if arr[i] == 0:
            out += "。"
            has_end = True
            break
        out += vocab[str(arr[i])]
    if not has_end:
        out += "。"
    txt = out.strip()

    w = jieba.cut(txt, cut_all=False)
    p = ' '.join(w)

    return p


def get_reward_cirder(gen_result, gts_data, opt):
    global CiderD_scorer
    if CiderD_scorer is None:
        # type = 0
        # if type == 0:
        #     path_cider = "/media/amds/data/code/cider"
        #     path_idxs = "/media/amds/data/dataset/mscoco"
        # else:
        #     path_cider = "/home/scw4750/caption/cider"
        #     path_idxs = "/home/scw4750/caption/dataset/mscoco"

        path_cider = opt.path_cider
        path_idxs = opt.path_idxs

        # /home/scw4750/caption/cider
        # /media/amds/data/code/cider
        sys.path.append(path_cider)
        from pyciderevalcap.ciderD.ciderD import CiderD

        # /home/scw4750/caption/dataset/mscoco
        # /media/amds/data/dataset/mscoco
        CiderD_scorer = CiderD(df='coco-train-idxs', path=path_idxs)

    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(gts_data)

    res = OrderedDict()
    gen_result = gen_result.cpu().numpy()

    # sample result
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(gts_data)):
        gts[i] = [array_to_str(gts_data[i][j]) for j in range(len(gts_data[i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]

    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}

    _, scores = CiderD_scorer.compute_score(gts, res)
    sample_mean = np.mean(scores)
    print('Cider scores: {:.3f} sample:{:.3f}'.format(_, sample_mean))

    # diff_result = sample_result - greedy_result
    # batch_size
    # scores = scores[:batch_size] - scores[batch_size:]

    # batch_size * seq_length
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards, sample_mean


def get_self_critical_reward(greedy_res, gen_result, gts_data, alpha, opt):
    global CiderD_scorer
    if CiderD_scorer is None:
        # type = 0
        # if type == 0:
        #     path_cider = "/media/amds/data/code/cider"
        #     path_idxs = "/media/amds/data/dataset/mscoco"
        # else:
        #     path_cider = "/home/scw4750/caption/cider"
        #     path_idxs = "/home/scw4750/caption/dataset/mscoco"

        path_cider = opt.path_cider
        path_idxs = opt.path_idxs

        # /home/scw4750/caption/cider
        # /media/amds/data/code/cider
        sys.path.append(path_cider)
        from pyciderevalcap.ciderD.ciderD import CiderD

        # /home/scw4750/caption/dataset/mscoco
        # /media/amds/data/dataset/mscoco
        print("load cider")
        CiderD_scorer = CiderD(df=opt.cider_idxs, path=path_idxs)

    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(gts_data)

    res = OrderedDict()
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()

    # sample result
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    # greedy result
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(gts_data)):
        gts[i] = [array_to_str(gts_data[i][j]) for j in range(len(gts_data[i]))]


    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    _, scores = CiderD_scorer.compute_score(gts, res)
    sample_mean = np.mean(scores[:batch_size])
    greedy_mean = np.mean(scores[batch_size:])
    print('Cider scores: {:.3f} sample:{:.3f} greedy:{:.3f}'.format(_, sample_mean, greedy_mean))

    # diff_result = sample_result - greedy_result
    # batch_size
    scores = scores[:batch_size] - scores[batch_size:] * alpha

    # batch_size * seq_length
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards, sample_mean, greedy_mean


def get_self_critical_reward_aic(greedy_res, gen_result, gts_data, alpha, vocab, opt):
    global CiderD_scorer
    global Bleu_scorer
    global Rouge_scorer
    if CiderD_scorer is None:
        # type = 0
        # if type == 0:
        #     path_cider = "/media/amds/data/code/cider"
        #     path_idxs = "/media/amds/data/dataset/mscoco"
        # else:
        #     path_cider = "/home/scw4750/caption/cider"
        #     path_idxs = "/home/scw4750/caption/dataset/mscoco"

        path_cider = opt.path_cider
        path_idxs = opt.path_idxs

        # /home/scw4750/caption/cider
        # /media/amds/data/code/cider
        sys.path.append(path_cider)
        from pyciderevalcap.ciderD.ciderD import CiderD
        from pyciderevalcap.bleu.bleu import Bleu
        from pyciderevalcap.rouge.rouge import Rouge
        from pyciderevalcap.meteor.meteor import Meteor

        # /home/scw4750/caption/dataset/mscoco
        # /media/amds/data/dataset/mscoco
        CiderD_scorer = CiderD(df=opt.cider_idxs, path=path_idxs)
        Bleu_scorer = Bleu()
        Rouge_scorer = Rouge()
        Meteor_scorer = Meteor()


    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(gts_data)

    res = OrderedDict()
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()

    # sample result
    for i in range(batch_size):
        res[i] = [array_to_str_aic(gen_result[i], vocab)]

    # greedy result
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str_aic(greedy_res[i], vocab)]

    gts = OrderedDict()
    for i in range(len(gts_data)):
        gts[i] = [array_to_str_aic(gts_data[i][j], vocab) for j in range(len(gts_data[i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    if opt.rl_metric == 'CIDEr':
        _, scores = CiderD_scorer.compute_score(gts, res)
    elif opt.rl_metric == 'ROUGE_L':
        _, scores = Rouge_scorer.compute_score(gts, res)
    elif opt.rl_metric == 'Bleu_4':
        _, scores = Bleu_scorer.compute_score(gts, res)
        _ = _[-1]
        scores = np.array(scores[-1])
    elif opt.rl_metric == 'AVG':
        d_, d_scores = CiderD_scorer.compute_score(gts, res)
        b_, b_scores = Bleu_scorer.compute_score(gts, res)
        r_, r_scores = Rouge_scorer.compute_score(gts, res)

        b_ = b_[-1]
        b_scores = np.array(b_scores[-1])

        _ = (d_ + b_ + r_)/3
        scores = (d_scores + b_scores + r_scores)/3
    elif opt.rl_metric == 'Meteor':
        _, scores = Meteor_scorer.compute_score(gts, res)


    sample_mean = np.mean(scores[:batch_size])
    greedy_mean = np.mean(scores[batch_size:])
    print('scores: {:.3f} sample:{:.3f} greedy:{:.3f}'.format(_, sample_mean, greedy_mean))

    # diff_result = sample_result - greedy_result
    # batch_size
    scores = scores[:batch_size] - scores[batch_size:] * alpha

    # batch_size * seq_length
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards, sample_mean, greedy_mean

def get_sample_reward_aic(sample_res, gts_data, gamma, vocab, opt):

    batch_size = sample_res.size(0)
    seq_length = sample_res.size(1)

    global CiderD_scorer
    global Bleu_scorer
    global Rouge_scorer
    if CiderD_scorer is None:
        # type = 0
        # if type == 0:
        #     path_cider = "/media/amds/data/code/cider"
        #     path_idxs = "/media/amds/data/dataset/mscoco"
        # else:
        #     path_cider = "/home/scw4750/caption/cider"
        #     path_idxs = "/home/scw4750/caption/dataset/mscoco"

        path_cider = opt.path_cider
        path_idxs = opt.path_idxs

        # /home/scw4750/caption/cider
        # /media/amds/data/code/cider
        sys.path.append(path_cider)
        from pyciderevalcap.ciderD.ciderD import CiderD
        from pyciderevalcap.bleu.bleu import Bleu
        from pyciderevalcap.rouge.rouge import Rouge
        from pyciderevalcap.meteor.meteor import Meteor

        # /home/scw4750/caption/dataset/mscoco
        # /media/amds/data/dataset/mscoco
        CiderD_scorer = CiderD(df=opt.cider_idxs, path=path_idxs)
        Bleu_scorer = Bleu()
        Rouge_scorer = Rouge()
        Meteor_scorer = Meteor()


    batch_size = sample_res.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(gts_data)

    res = OrderedDict()
    sample_res = sample_res.cpu().numpy()

    # sample result
    for i in range(batch_size):
        res[i] = [array_to_str_aic(sample_res[i], vocab)]

    gts = OrderedDict()
    for i in range(len(gts_data)):
        gts[i] = [array_to_str_aic(gts_data[i][j], vocab) for j in range(len(gts_data[i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}

    if opt.rl_metric == 'CIDEr':
        _, scores = CiderD_scorer.compute_score(gts, res)
    elif opt.rl_metric == 'ROUGE_L':
        _, scores = Rouge_scorer.compute_score(gts, res)
    elif opt.rl_metric == 'Bleu_4':
        _, scores = Bleu_scorer.compute_score(gts, res)
        _ = _[-1]
        scores = np.array(scores[-1])
    elif opt.rl_metric == 'AVG':
        d_, d_scores = CiderD_scorer.compute_score(gts, res)
        b_, b_scores = Bleu_scorer.compute_score(gts, res)
        r_, r_scores = Rouge_scorer.compute_score(gts, res)

        b_ = b_[-1]
        b_scores = np.array(b_scores[-1])

        _ = (d_ + b_ + r_)/3
        scores = (d_scores + b_scores + r_scores)/3
    elif opt.rl_metric == 'Meteor':
        _, scores = Meteor_scorer.compute_score(gts, res)

    # sample batch
    sample_mean = np.mean(scores)
    print('scores: {:.3f} sample:{:.3f}'.format(_, sample_mean))

    # batch_size
    sample_reward = scores

    # seq_length
    list_gamma = np.logspace(seq_length-1, 0, seq_length, base=gamma)
    # batch_size * seq_length
    batch_gamma = np.repeat(list_gamma[np.newaxis, :], batch_size, 0)
    # batch_size * seq_length
    batch_sample_reward = np.repeat(sample_reward[:, np.newaxis], seq_length, 1)

    # batch_size * (seq_length+1)
    full_sample_reward = batch_gamma * batch_sample_reward

    # sample_reward : batch_size
    # sample_mean : 1
    # full_sample_reward : batch_size * (seq_length+1)
    return full_sample_reward, sample_mean


def get_reward(inputs, targets, type):

    gts = {}
    res = {}

    for i in xrange(len(inputs)):
        gts[i] = []
        gts[i].append(targets[i])
        res[i] = []
        res[i].append(inputs[i])

    score = 0
    if type == "CIDEr":
        scorer = Cider()
        score, scores = scorer.compute_score(gts, res)
    elif type == "Bleu":
        scorer = Bleu(4)
        score, scores = scorer.compute_score(gts, res)

    return score, scores

def get_reward1(input, target, type):

    gts = {}
    res = {}

    for i in xrange(1000,1001):
        gts[i] = []
        gts[i].append(target)
        res[i] = []
        res[i].append(input)

    score = 0
    if type == "CIDEr":
        scorer = Cider()
        score, scores = scorer.compute_score(gts, res)
    elif type == "Bleu":
        scorer = Bleu(4)
        score, scores = scorer.compute_score(gts, res)

    return score, scores

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=0)

def prepro(imgs, data_augment):

    h, w = imgs.size(2), imgs.size(3)

    cnn_input_size = 224

    imgs_new = torch.zeros((imgs.size(0),imgs.size(1),cnn_input_size,cnn_input_size)).type_as(imgs)

    for i in xrange(imgs.size(0)):

        if data_augment:
            rx = np.random.randint(w - cnn_input_size)
            ry = np.random.randint(h - cnn_input_size)
        else:
            rx = (w - cnn_input_size) // 2
            ry = (h - cnn_input_size) // 2

        imgs_new[i] = imgs[i, :, rx:rx+cnn_input_size, ry:ry+cnn_input_size]

    vgg_mean = torch.FloatTensor([-123.68, -116.779, -103.939]).cuda().view(1, 3, 1, 1)
    vgg_mean = vgg_mean.expand_as(imgs_new)
    imgs_new.add_(vgg_mean)

    return imgs_new

def prepro_norm(imgs, data_augment):

    h, w = imgs.size(2), imgs.size(3)

    cnn_input_size = 224

    imgs_new = torch.zeros((imgs.size(0),imgs.size(1),cnn_input_size,cnn_input_size)).type_as(imgs)

    for i in xrange(imgs.size(0)):

        if data_augment:
            if w == cnn_input_size:
                rx = 0
            else:
                rx = np.random.randint(w - cnn_input_size)

            if h == cnn_input_size:
                ry = 0
            else:
                ry = np.random.randint(h - cnn_input_size)
        else:
            rx = (w - cnn_input_size) // 2
            ry = (h - cnn_input_size) // 2

        imgs_new[i] = imgs[i, :, rx:rx+cnn_input_size, ry:ry+cnn_input_size]

    # range [0.0 ~ 1.0]
    imgs_new.div_(255.0)

    #
    vgg_mean = torch.FloatTensor([-0.485, -0.456, -0.406]).cuda().view(1, 3, 1, 1)
    vgg_mean = vgg_mean.expand_as(imgs_new)
    imgs_new.add_(vgg_mean)

    vgg_std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
    vgg_std = vgg_std.expand_as(imgs_new)
    imgs_new.div_(vgg_std)

    return imgs_new


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

if __name__ == "__main__":
    sent = ["a crowd has gathered to board a passenger jet","a cell phone and a watch on a table"]
    ref = ["a crowd has gathered to board a passenger jet","a cell phone a watch on a table"]
    reward = get_reward(sent, ref, "CIDEr")
    print(reward)