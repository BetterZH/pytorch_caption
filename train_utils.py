from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import os
import time

import torch.optim as optim
from data.DataLoaderThread import *
from data.DataLoaderThreadBu import *
from data.DataLoaderThreadNew import *

import eval_utils
import misc.Criterion as Criterion
import models
import opts
import tensorboard
from data.DataLoader import *
import torch.nn as nn
import logger.logger as logger
from notify.notify import notify
import trans_client


class train_step:
    def __init__(self):
        self.learning_rate = 4e-4
        self.cnn_learning_rate = 1e-5
        self.finetune_cnn_after = -1

def init_train_process():

    train_process = []

    step1 = train_step()
    step1.learning_rate = 4e-4
    step1.cnn_learning_rate = 5e-5
    step1.finetune_cnn_after = -1

    step2 = train_step()
    step2.learning_rate = 1e-4
    step2.cnn_learning_rate = 5e-5
    step2.finetune_cnn_after = -1

    step3 = train_step()
    step3.learning_rate = 4e-5
    step3.cnn_learning_rate = 5e-5
    step3.finetune_cnn_after = -1

    step4 = train_step()
    step4.learning_rate = 4e-5
    step4.cnn_learning_rate = 5e-5
    step4.finetune_cnn_after = 0

    step5 = train_step()
    step5.learning_rate = 4e-5
    step5.cnn_learning_rate = 1e-5
    step5.finetune_cnn_after = 0

    step6 = train_step()
    step6.learning_rate = 4e-5
    step6.cnn_learning_rate = 1e-6
    step6.finetune_cnn_after = 0

    train_process.append(step1)
    train_process.append(step2)
    train_process.append(step3)
    train_process.append(step4)
    train_process.append(step5)
    train_process.append(step6)


    return train_process



def load_infos(opt):
    infos = {}
    start_from_best = opt.start_from_best.strip()
    start_from = opt.start_from.strip()
    if start_from_best is not None and len(start_from_best) > 0:
        print("start best from %s" % (start_from_best))
        # open old infos and check if models are compatible
        with open(os.path.join(start_from_best, opt.id + '_infos_best.pkl')) as f:
            infos = cPickle.load(f)
    elif start_from is not None and len(start_from) > 0:
        print("start from %s" % (start_from))
        # open old infos and check if models are compatible
        with open(os.path.join(start_from, opt.id + '_infos.pkl')) as f:
            infos = cPickle.load(f)
    return infos

def save_model_conf(model_cnn, model, opt):
    model_cnn_file = os.path.join(opt.eval_result_path, opt.id + '_model_cnn.txt')
    with open(model_cnn_file, 'w') as f:
        f.write("{}\n".format(model_cnn))

    model_file = os.path.join(opt.eval_result_path, opt.id + '_model.txt')
    with open(model_file, 'w') as f:
        f.write("{}\n".format(model))


def train_reinforce(params, opt):

    model = params['model']
    fc_feats = params['fc_feats']
    att_feats = params['att_feats']
    labels = params['labels']
    masks = params['masks']
    vocab = params['vocab']
    crit_pg = params['crit_pg']
    crit_rl = params['crit_rl']
    targets = params['targets']
    gts = params['gts']

    if models.has_bu(opt.caption_model) or models.has_sub_region_bu(opt.caption_model):
        bu_feats = params['bu_feats']

    # compute policy gradient
    if opt.reinforce_type == 0:
        raise Exception('reinforce_type error, 0 is deprecated')
        # forward
        start = time.time()
        if models.is_only_fc_feat(opt.caption_model):
            output = model(fc_feats, labels)
        else:
            output = model(fc_feats, att_feats, labels)

        if opt.verbose:
            print('model {:.3f}'.format(time.time() - start))

        train_loss, reward_mean = crit_pg.forward_backward(output, labels, masks, vocab)
    # self-critical
    elif opt.reinforce_type == 1:
        # forward
        start = time.time()
        if models.is_only_fc_feat(opt.caption_model):
            sample_seq, sample_seqLogprobs = model.sample(fc_feats, {'sample_max': 0})
            greedy_seq, greedy_seqLogprobs = model.sample(fc_feats, {'sample_max': 1})
        elif models.is_only_att_feat(opt.caption_model):
            sample_seq, sample_seqLogprobs = model.sample(att_feats, {'sample_max': 0})
            greedy_seq, greedy_seqLogprobs = model.sample(att_feats, {'sample_max': 1})
        elif models.has_bu(opt.caption_model) or models.has_sub_region_bu(opt.caption_model):
            sample_seq, sample_seqLogprobs = model.sample(fc_feats, att_feats, bu_feats, {'sample_max': 0})
            greedy_seq, greedy_seqLogprobs = model.sample(fc_feats, att_feats, bu_feats, {'sample_max': 1})
        else:
            # sample_seq, sample_seqLogprobs = model.sample_forward(fc_feats, att_feats, labels, {'sample_max': 0})
            # greedy_seq, greedy_seqLogprobs = model.sample_forward(fc_feats, att_feats, labels, {'sample_max': 1})
            sample_output = model.sample(fc_feats, att_feats, {'sample_max': 0})
            greedy_output = model.sample(fc_feats, att_feats, {'sample_max': 1})

            sample_seq = sample_output[0]
            sample_seqLogprobs = sample_output[1]

            greedy_seq = greedy_output[0]
            greedy_seqLogprobs = greedy_output[1]

        if opt.verbose:
            print('model {:.3f}'.format(time.time() - start))

        # compute the loss
        start = time.time()
        # seq, seqLogprobs, seq1, target, vocab
        loss, reward_mean, sample_mean, greedy_mean = crit_rl(sample_seq, sample_seqLogprobs, greedy_seq, gts, masks)
        # loss, reward_mean = crit_rl(sample_seq, sample_seqLogprobs, gts)
        if opt.verbose:
            print('crit {:.3f}'.format(time.time() - start))

        # backward
        start = time.time()
        loss.backward()
        if opt.verbose:
            print('loss {:.3f}'.format(time.time() - start))

        # show information
        train_loss = loss.data[0]

    return train_loss, reward_mean, sample_mean, greedy_mean

# type 0: only critic
# type 1: critic-actor
def train_actor_critic(params, opt, type, retain_graph=False):

    model = params['model']
    fc_feats = params['fc_feats']
    att_feats = params['att_feats']
    labels = params['labels']
    masks = params['masks']
    vocab = params['vocab']
    gts = params['gts']

    if type == 0:
        crit_c = params['crit_c']
    elif type == 1:
        crit_ac = params['crit_ac']

    if models.has_bu(opt.caption_model) or models.has_sub_region_bu(opt.caption_model):
        bu_feats = params['bu_feats']

    # forward
    start = time.time()
    if models.is_only_fc_feat(opt.caption_model):
        sample_seq, sample_seqLogprobs, sample_value = model.sample(fc_feats, {'sample_max': 0})
    elif models.has_bu(opt.caption_model) or models.has_sub_region_bu(opt.caption_model):
        sample_seq, sample_seqLogprobs, sample_value = model.sample(fc_feats, att_feats, bu_feats, {'sample_max': 0})
    else:
        # sample_seq, sample_seqLogprobs = model.sample_forward(fc_feats, att_feats, labels, {'sample_max': 0})
        # greedy_seq, greedy_seqLogprobs = model.sample_forward(fc_feats, att_feats, labels, {'sample_max': 1})
        sample_output = model.sample(fc_feats, att_feats, {'sample_max': 0})

        sample_seq = sample_output[0]
        sample_seqLogprobs = sample_output[1]
        sample_value = sample_output[2]

    if opt.verbose:
        print('model {:.3f}'.format(time.time() - start))

    # compute the loss
    start = time.time()
    # 0. critic
    # 1. critic, actor
    if type == 0:
        # seq, seqLogprobs, seq1, target, vocab
        loss, reward_mean, sample_mean = crit_c(sample_seq, sample_value, gts)
    elif type == 1:
        # seq, seqLogprobs, seq1, target, vocab
        loss, reward_mean, sample_mean = crit_ac(sample_seq, sample_seqLogprobs, sample_value, gts)
    # loss, reward_mean = crit_rl(sample_seq, sample_seqLogprobs, gts)
    if opt.verbose:
        print('crit {:.3f}'.format(time.time() - start))

    # backward
    start = time.time()
    loss.backward(retain_graph=retain_graph)
    if opt.verbose:
        print('loss {:.3f}'.format(time.time() - start))

    # show information
    train_loss = loss.data[0]

    return train_loss, reward_mean, sample_mean


def train_mix(params, iteration, opt):

    model = params['model']
    fc_feats = params['fc_feats']
    att_feats = params['att_feats']
    labels = params['labels']
    masks = params['masks']
    vocab = params['vocab']
    gts = params['gts']
    crit_pg = params['crit_pg']
    crit = params['crit']

    output = None

    if iteration % 2 == 1:
        use_reinforce = True
        train_loss, reward_mean = crit_pg.forward_backward(output, labels, masks, vocab)
    else:
        use_reinforce = False

        # forward
        start = time.time()
        if models.is_only_fc_feat(opt.caption_model):
            output = model(fc_feats, labels)
        else:
            output = model(fc_feats, att_feats, labels)

        if opt.verbose:
            print('model {:.3f}'.format(time.time() - start))

        # compute the loss
        start = time.time()
        loss = crit(output, labels, masks)
        if opt.verbose:
            print('crit {:.3f}'.format(time.time() - start))

        # backward
        start = time.time()
        loss.backward()
        if opt.verbose:
            print('loss {:.3f}'.format(time.time() - start))

        # show information
        train_loss = loss.data[0]
        reward_mean = 0
    return train_loss, reward_mean

def train_normal(params, opt):

    model = params['model']
    fc_feats = params['fc_feats']
    att_feats = params['att_feats']
    labels = params['labels']
    targets = params['targets']
    masks = params['masks']
    vocab = params['vocab']
    crit = params['crit']

    # forward
    start = time.time()
    if models.is_transformer(opt.caption_model):
        output = model(att_feats, targets, masks)
    elif models.is_ctransformer(opt.caption_model):
        output = model(fc_feats, att_feats, targets, masks)
    elif models.is_only_fc_feat(opt.caption_model):
        output = model(fc_feats, labels)
    elif models.is_only_att_feat(opt.caption_model):
        output = model(att_feats, labels)
    elif models.has_bu(opt.caption_model):
        bu_feats = params['bu_feats']
        output = model(fc_feats, att_feats, bu_feats, labels)
    else:
        output = model(fc_feats, att_feats, labels)

    if opt.verbose:
        print('model {:.3f}'.format(time.time() - start))

    # compute the loss
    start = time.time()

    if models.is_prob_weight(opt.caption_model):
        output = output[0]

    loss = crit(output, labels, masks)
    if opt.verbose:
        print('crit {:.3f}'.format(time.time() - start))

    # backward
    start = time.time()
    loss.backward()
    if opt.verbose:
        print('loss {:.3f}'.format(time.time() - start))

    # show information
    train_loss = loss.data[0]
    reward_mean = 0

    return train_loss, reward_mean

def train_value(params, opt):

    model = params['model']
    fc_feats = params['fc_feats']
    att_feats = params['att_feats']
    labels = params['labels']
    targets = params['targets']
    masks = params['masks']
    crit = params['crit']

    # forward
    start = time.time()
    if models.is_transformer(opt.caption_model):
        output = model(att_feats, targets, masks)
    elif models.is_ctransformer(opt.caption_model):
        output = model(fc_feats, att_feats, targets, masks)
    elif models.is_only_fc_feat(opt.caption_model):
        output = model(fc_feats, labels)
    elif models.is_only_att_feat(opt.caption_model):
        output = model(att_feats, labels)
    elif models.has_bu(opt.caption_model):
        bu_feats = params['bu_feats']
        output = model(fc_feats, att_feats, bu_feats, labels)
    else:
        output = model(fc_feats, att_feats, labels)

    if opt.verbose:
        print('model {:.3f}'.format(time.time() - start))

    # compute the loss
    start = time.time()

    if models.is_prob_weight(opt.caption_model):
        output = output[0]

    loss = crit(output, labels, masks)
    if opt.verbose:
        print('crit {:.3f}'.format(time.time() - start))

    # backward
    start = time.time()
    loss.backward()
    if opt.verbose:
        print('loss {:.3f}'.format(time.time() - start))

    # show information
    train_loss = loss.data[0]
    reward_mean = 0

    return train_loss, reward_mean


def train_with_prob_weight(params, opt):

    model = params['model']
    fc_feats = params['fc_feats']
    att_feats = params['att_feats']
    labels = params['labels']
    targets = params['targets']
    masks = params['masks']
    tokens = params['tokens']
    crit = params['crit']

    # forward
    start = time.time()

    if models.is_transformer(opt.caption_model):
        output, prob_w = model(att_feats, targets, masks)
    elif models.is_ctransformer(opt.caption_model):
        output, prob_w = model(fc_feats, att_feats, targets, masks)
    elif models.has_bu(opt.caption_model) or models.has_sub_region_bu(opt.caption_model):
        bu_feats = params['bu_feats']
        output, prob_w = model(fc_feats, att_feats, bu_feats, labels)
    else:
        output, prob_w = model(fc_feats, att_feats, labels)

    if opt.verbose:
        print('model {:.3f}'.format(time.time() - start))

    # compute the loss
    start = time.time()
    # input, target, mask, prob_w, token, alpha)
    loss = crit(output, labels, masks, prob_w, tokens)
    if opt.verbose:
        print('crit {:.3f}'.format(time.time() - start))

    # backward
    start = time.time()
    loss.backward()
    if opt.verbose:
        print('loss {:.3f}'.format(time.time() - start))

    # show information
    train_loss = loss.data[0]
    reward_mean = 0

    return train_loss, reward_mean


def save_result(str_stats, predictions, opt):
    eval_result_file = os.path.join(opt.eval_result_path, opt.id + ".csv")
    with open(eval_result_file, 'a') as f:
        f.write(str_stats + "\n")

    predictions_file = os.path.join(opt.eval_result_path, opt.id + ".json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

def save_best_result(predictions, opt):
    predictions_file = os.path.join(opt.eval_result_path, opt.id + "_best.json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

def save_model_iteration(model, model_cnn, infos, opt, iteration):

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    checkpoint_path = os.path.join(opt.checkpoint_path, opt.id + '_model_' + str(iteration) + '.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    checkpoint_path_cnn = os.path.join(opt.checkpoint_path, opt.id + '_model_cnn_' + str(iteration) + '.pth')
    torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
    print("model cnn saved to {}".format(checkpoint_path_cnn))

    info_path = os.path.join(opt.checkpoint_path, opt.id + '_infos_' + str(iteration) + '.pkl')
    with open(info_path, 'wb') as f:
        cPickle.dump(infos, f)

def save_model(model, model_cnn, infos, opt):

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    checkpoint_path = os.path.join(opt.checkpoint_path, opt.id + '_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    checkpoint_path_cnn = os.path.join(opt.checkpoint_path, opt.id + '_model_cnn.pth')
    torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
    print("model cnn saved to {}".format(checkpoint_path_cnn))

    info_path = os.path.join(opt.checkpoint_path, opt.id + '_infos.pkl')
    with open(info_path, 'wb') as f:
        cPickle.dump(infos, f)

def save_infos(infos, opt):
    info_path = os.path.join(opt.checkpoint_path, opt.id + '_infos.pkl')
    with open(info_path, 'wb') as f:
        cPickle.dump(infos, f)

def save_model_best(model, model_cnn, infos, opt):
    if not os.path.exists(opt.checkpoint_best_path):
        os.makedirs(opt.checkpoint_best_path)

    checkpoint_path = os.path.join(opt.checkpoint_best_path, opt.id + '_model_best.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    checkpoint_path_cnn = os.path.join(opt.checkpoint_best_path, opt.id + '_model_cnn_best.pth')
    torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
    print("model cnn saved to {}".format(checkpoint_path_cnn))

    info_path = os.path.join(opt.checkpoint_best_path, opt.id + '_infos_best.pkl')
    with open(info_path, 'wb') as f:
        cPickle.dump(infos, f)

    # save all model
    # checkpoint_path = os.path.join(opt.checkpoint_best_path, 'model_' + opt.id + '_best_all.pth')
    # torch.save(model, checkpoint_path)
    # print("model saved to {}".format(checkpoint_path))

    # checkpoint_path_cnn = os.path.join(opt.checkpoint_best_path, 'all_model_cnn_' + opt.id + '_best_all.pth')
    # torch.save(model_cnn, checkpoint_path_cnn)
    # print("model cnn saved to {}".format(checkpoint_path_cnn))

def get_cnn_optimizer(model_cnn, optimizer_cnn, finetune_cnn_start, opt):
    if finetune_cnn_start:
        if optimizer_cnn is None:
            if opt.finetune_cnn_type == 0:
                for p in model_cnn.parameters():
                    p.requires_grad = True
                params = model_cnn.parameters()
            elif opt.finetune_cnn_type == 1:
                for p in model_cnn.parameters():
                    p.requires_grad = False
                layers = models.get_fintune_layers(model_cnn, opt)
                params = []
                for layer in layers:
                    for p in layer.parameters():
                        p.requires_grad = True
                    params.append({'params': layer.parameters()})
            else:
                raise Exception('finetune_cnn_type error')
            optimizer_cnn = optim.Adam(params, lr=opt.cnn_learning_rate, betas=(opt.optim_alpha, opt.optim_beta),
                                       eps=opt.optim_epsilon, weight_decay=opt.cnn_weight_decay)
        model_cnn.train()
    else:
        for p in model_cnn.parameters():
            p.requires_grad = False
        model_cnn.eval()

    return optimizer_cnn

def update_lr(epoch, optimizer, optimizer_cnn, finetune_cnn_start, opt):
    # Assign the learning rate
    if opt.learning_rate_decay_start >= 0 and epoch >= opt.learning_rate_decay_start:
        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
        utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
    else:
        opt.current_lr = opt.learning_rate

    if finetune_cnn_start and opt.cnn_learning_rate_decay_start >= 0 and epoch >= opt.cnn_learning_rate_decay_start:
        frac = (epoch - opt.cnn_learning_rate_decay_start) // opt.cnn_learning_rate_decay_every
        decay_factor = opt.cnn_learning_rate_decay_rate ** frac
        opt.current_cnn_lr = opt.cnn_learning_rate * decay_factor
        utils.set_lr(optimizer_cnn, opt.current_cnn_lr)  # set the decayed rate
    else:
        opt.current_cnn_lr = opt.cnn_learning_rate