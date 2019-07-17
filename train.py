from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import os
import time

import torch.optim as optim
from data.DataLoaderThreadBu import *
from data.DataLoaderThreadNew import *
from data.DataLoaderThreadRegion import *

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
import train_utils


def train_cnn(model_cnn, images, bus, fc_expander, att_expander, bu_expander, use_reinforce):

    fc_feats = None
    att_feats = None
    bu_feats = None

    # train cnn
    if models.is_only_fc_feat(opt.caption_model):
        fc_feats = model_cnn(images)
        if opt.seq_per_img > 1 and not use_reinforce:
            fc_feats = fc_expander(fc_feats)
    elif models.is_only_att_feat(opt.caption_model):
        att_feats = model_cnn(images)
        if opt.seq_per_img > 1 and not use_reinforce:
            att_feats = att_expander(att_feats)
    elif models.has_sub_region_bu(opt.caption_model):
        fc_feats, att_feats, bu_feats = model_cnn(images)
        if opt.seq_per_img > 1 and not use_reinforce:
            fc_feats = fc_expander(fc_feats)
            att_feats = att_expander(att_feats)
            bu_feats = bu_expander(bu_feats)
    else:
        fc_feats, att_feats = model_cnn(images)
        if opt.seq_per_img > 1 and not use_reinforce:
            fc_feats = fc_expander(fc_feats)
            att_feats = att_expander(att_feats)

    if models.has_bu(opt.caption_model):
        bus_feats = bus
        if opt.seq_per_img > 1 and not use_reinforce:
            bu_feats = bu_expander(bus_feats)

    return fc_feats, att_feats, bu_feats


# crit_pg, crit_rl, crit_ctc, crit_c, crit_ac, crit,
def train_model(params, iteration, epoch, board):

    vocab = params['vocab']

    if opt.reinforce_start >= 0 and epoch >= opt.reinforce_start:
        use_reinforce = True

        # create crit
        if params['crit_pg'] is None:
            params['crit_pg'] = Criterion.PGCriterion(opt)
        if params['crit_rl'] is None:
            if opt.is_aic_data:
                if models.is_prob_weight_mul_out(opt.caption_model):
                    crit_rl = Criterion.RewardMulOutCriterionAIC(opt, vocab)
                else:
                    crit_rl = Criterion.RewardCriterionAIC(opt, vocab)
            else:
                crit_rl = Criterion.RewardCriterion(opt)
            params['crit_rl'] = crit_rl

        train_loss, reward_mean, sample_mean, greedy_mean = train_utils.train_reinforce(params, opt)

        if opt.use_tensorboard:
            if iteration % opt.tensorboard_for_train_every == 0:
                board.val("sample_mean", sample_mean, iteration)
                board.val("greedy_mean", greedy_mean, iteration)

    elif opt.mix_start >= 0 and epoch >= opt.mix_start:
        raise Exception('mix is deprecated')
        if params['crit_pg'] is None:
            params['crit_pg'] = Criterion.PGCriterion(opt)
        if params['crit'] is None:
            params['crit'] = get_criterion()
        train_loss, reward_mean = train_utils.train_mix(params, iteration, opt)
    elif opt.ctc_start >= 0 and epoch >= opt.ctc_start:

        use_reinforce = False
        if params['crit_ctc'] is None:
            params['crit_ctc'] = Criterion.CTCCriterion()
        train_loss, reward_mean = train_utils.train_normal(params, opt)

    elif opt.rl_critic_start >= 0 and epoch >= opt.rl_critic_start:
        use_reinforce = True
        if params['crit_c'] is None:
            params['crit_c'] = Criterion.ActorCriticMSECriterionAIC(opt, vocab)
        train_loss, reward_mean, sample_mean = train_utils.train_actor_critic(params, opt, 0)
        if opt.use_tensorboard:
            if iteration % opt.tensorboard_for_train_every == 0:
                board.val("reward_mean", reward_mean, iteration)
                board.val("sample_mean", sample_mean, iteration)
    elif opt.rl_actor_critic_start >= 0 and epoch >= opt.rl_actor_critic_start:
        use_reinforce = True
        if params['crit_c'] is None:
            params['crit_c'] = Criterion.ActorCriticMSECriterionAIC(opt, vocab)
        if params['crit_ac'] is None:
            params['crit_ac'] = Criterion.ActorCriticCriterionAIC(opt, vocab)
        train_loss1, reward_mean1, sample_mean1 = train_utils.train_actor_critic(params, opt, 0, retain_graph=True)
        print("critic loss: {:.3f} reward_mean: {:.3f}  sample_mean: {:.3f} ".format(train_loss1, reward_mean1,
                                                                                     sample_mean1))
        train_loss, reward_mean, sample_mean = train_utils.train_actor_critic(params, opt, 1)
        print("actor critic loss: {:.3f} reward_mean: {:.3f}  sample_mean: {:.3f} ".format(train_loss, reward_mean,
                                                                                           sample_mean))
        if opt.use_tensorboard:
            if iteration % opt.tensorboard_for_train_every == 0:
                board.val("reward_mean", reward_mean, iteration)
                board.val("sample_mean", sample_mean, iteration)
    else:
        use_reinforce = False
        if params['crit'] is None:
            params['crit'] = get_criterion()
        if models.is_prob_weight(opt.caption_model) or models.is_prob_weight_mul_out(opt.caption_model):
            train_loss, reward_mean = train_utils.train_with_prob_weight(params, opt)
        else:
            train_loss, reward_mean = train_utils.train_normal(params, opt)

    return train_loss, reward_mean, use_reinforce

def get_criterion():
    # crit = Criterion.LanguageModelWeightNewCriterion()
    if models.is_mul_out_with_weight(opt.caption_model):
        crit = Criterion.LanguageModelWeightMulOutWithWeightCriterion(opt.prob_weight_alpha)
    elif models.is_mul_out(opt.caption_model):
        crit = Criterion.LanguageModelWeightMulOutCriterion()
    elif models.is_prob_weight(opt.caption_model):
        crit = Criterion.LanguageModelWithProbWeightCriterion(opt.prob_weight_alpha)
    elif models.is_prob_weight_mul_out(opt.caption_model):
        crit = Criterion.LanguageModelWithProbWeightMulOutCriterion(opt.prob_weight_alpha)
    else:
        crit = Criterion.LanguageModelWeightCriterion()
    return crit

def get_loader():
    if models.has_bu(opt.caption_model) or \
            models.has_sub_regions(opt.caption_model) or \
            models.has_sub_region_bu(opt.caption_model):
        loader = DataLoaderThreadBu(opt)
        print("DataLoaderThreadBu")
    else:
        loader = DataLoaderThreadNew(opt)
        print("DataLoaderThreadNew")
    return loader

def get_infos():
    try:
        if opt.is_load_infos == 1:
            infos = train_utils.load_infos(opt)
        else:
            infos = {}
    except:
        infos = {}
        print('load infos error')
    return infos

def get_expander():
    fc_expander = None
    att_expander = None
    bu_expander = None
    if opt.seq_per_img > 1:
        fc_expander = utils.FeatExpander(opt.seq_per_img)
        att_expander = utils.FeatExpander(opt.seq_per_img)
        if models.has_bu(opt.caption_model) or models.has_sub_region_bu(opt.caption_model):
            bu_expander = utils.FeatExpander(opt.seq_per_img)
    return fc_expander, att_expander, bu_expander

def eval_model(model_cnn, model, params, loader, board, iteration, notifier, val_result_history, best_val_score):

    if opt.is_every_eval:

        # eval model
        eval_kwargs = {'split': opt.val_split,
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        print("start eval ...")

        val_loss, predictions, lang_stats, str_stats = eval_utils.eval_split(model_cnn, model, params['crit'], loader,
                                                                             eval_kwargs)

        if opt.use_tensorboard:
            board.accuracy(lang_stats, iteration)
            board.loss_val(val_loss, iteration)

        print("end eval ...")

        msg = "iteration = {} val_loss = {} str_stats = {}".format(iteration, val_loss, str_stats)
        notifier.send(opt.id + " val result", opt.id + " :\n" + msg)
        logger.info(msg)

        train_utils.save_result(str(iteration) + "," + str_stats, predictions, opt)

        val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                         'predictions': predictions}

        # Save model if is improving on validation result
        if opt.language_eval == 1:
            eval_metric = opt.eval_metric
            current_score = lang_stats[eval_metric]
        else:
            current_score = - val_loss

        best_flag = False
        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            best_flag = True

    else:
        best_flag = True

    return predictions, best_val_score, best_flag, current_score

def get_optimizer(optimizer, epoch, model, model_cnn):
    if optimizer is None:
        if opt.rl_critic_start >= 0 and epoch >= opt.rl_critic_start:
            for param in model.parameters():
                param.requires_grad = False
            for param in model_cnn.parameters():
                param.requires_grad = False
            for param in model.value.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.value.parameters(), lr=opt.learning_rate,
                                   betas=(opt.optim_alpha, opt.optim_beta),
                                   eps=opt.optim_epsilon)
        else:
            # if models.is_transformer(opt.caption_model) or models.is_ctransformer(opt.caption_model):
            #     parameters = model.get_trainable_parameters()
            # else:
            #     parameters = model.parameters()
            parameters = model.parameters()
            optimizer = optim.Adam(parameters, lr=opt.learning_rate,
                                   betas=(opt.optim_alpha, opt.optim_beta),
                                   eps=opt.optim_epsilon)
    return optimizer

def update_gradient(optimizer, optimizer_cnn, finetune_cnn_start):
    start = time.time()
    utils.clip_gradient(optimizer, opt.grad_clip)
    optimizer.step()
    if opt.verbose:
        print('model_update {:.3f}'.format(time.time() - start))
    start = time.time()
    if finetune_cnn_start:
        utils.clip_gradient(optimizer_cnn, opt.grad_clip)
        optimizer_cnn.step()
        if opt.verbose:
            print('model_cnn_update {:.3f}'.format(time.time() - start))

def train(opt):

    notifier = notify()
    notifier.login()

    # init path
    if not os.path.exists(opt.eval_result_path):
        os.makedirs(opt.eval_result_path)

    config_file = os.path.join(opt.eval_result_path, opt.id + '_config.txt')
    with open(config_file, 'w') as f:
        f.write("{}\n".format(json.dumps(vars(opt), sort_keys=True, indent=2)))

    torch.backends.cudnn.benchmark = True

    if opt.use_tensorboard:

        if opt.tensorboard_type == 0:
            board = tensorboard.TensorBoard()
            board.start(opt.id, opt.tensorboard_ip, opt.tensorboard_port)
        else:
            board = trans_client.TransClient()
            board.start(opt.id)

    print(opt.cnn_model)

    loader = get_loader()

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    vocab = loader.get_vocab()
    opt.vocab = vocab
    batch_size = loader.batch_size

    infos = get_infos()
    infos['vocab'] = vocab

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    finetune_cnn_history = infos.get('finetune_cnn_history', {})


    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score = None

    model_cnn = models.setup_cnn(opt)
    model_cnn = model_cnn.cuda()
    model_cnn = nn.DataParallel(model_cnn)


    model = models.setup(opt)
    model = model.cuda()
    # if models.is_transformer(opt.caption_model) or models.is_ctransformer(opt.caption_model):
    #     model = nn.DataParallel(model)

    train_utils.save_model_conf(model_cnn, model, opt)

    update_lr_flag = True

    model_cnn.train()
    model.train()

    fc_expander, att_expander, bu_expander = get_expander()

    optimizer = None
    optimizer_cnn = None
    finetune_cnn_start = False

    early_stop_cnt = 0

    params = {}
    params['model'] = model
    params['vocab'] = vocab

    # crit_pg, crit_rl, crit_ctc, crit_c, crit_ac, crit
    params['crit_pg'] = None
    params['crit_rl'] = None
    params['crit_ctc'] = None
    params['crit_c'] = None
    params['crit_ac'] = None
    params['crit'] = None

    is_eval_start = opt.is_eval_start

    if opt.use_auto_learning_rate == 1:
        train_process = train_utils.init_train_process()
        train_process_index = infos.get('train_process_index', 0)
        train_step = train_process[train_process_index]
        optimizer_cnn = None
        optimizer = None
        opt.learning_rate = train_step.learning_rate
        opt.cnn_learning_rate = train_step.cnn_learning_rate
        opt.finetune_cnn_after = train_step.finetune_cnn_after

    while True:

        current_score = None

        # make evaluation on validation set, and save model
        if (iteration > 0 and iteration % opt.save_checkpoint_every == 0 and
                not val_result_history.has_key(iteration)) or is_eval_start:

            predictions, best_val_score, best_flag, current_score = eval_model(model_cnn, model, params,
                            loader, board, iteration, notifier, val_result_history, best_val_score)

            infos['best_val_score'] = best_val_score
            infos['val_result_history'] = val_result_history
            train_utils.save_infos(infos, opt)

            if best_flag:
                train_utils.save_best_result(predictions, opt)
                train_utils.save_model_best(model, model_cnn, infos, opt)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            is_eval_start = False

        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            msg = "max epoch"
            logger.info(msg)
            break

        # auto update model
        if opt.use_auto_learning_rate == 1 and current_score is not None:
            if early_stop_cnt > opt.auto_early_stop_cnt or current_score < opt.auto_early_stop_score:
                early_stop_cnt = 0
                train_process_index += 1
                msg = opt.id + " early stop " + str(train_process_index)
                logger.info(msg)

                infos['train_process_index'] = train_process_index
                train_utils.save_infos(infos, opt)

                if train_process_index >= len(train_process):
                    notifier.send(opt.id + " early stop", msg)
                    logger.info("break")
                    break

                train_step = train_process[train_process_index]
                optimizer_cnn = None
                optimizer = None
                opt.learning_rate = train_step.learning_rate
                opt.cnn_learning_rate = train_step.cnn_learning_rate
                opt.finetune_cnn_after = train_step.finetune_cnn_after
                opt.start_from_best = opt.auto_start_from_best

                # model_cnn_path = os.path.join(opt.auto_start_from_best, opt.id + '_model_cnn_best.pth')
                # model_cnn.load_state_dict(torch.load(model_cnn_path))
                # model_cnn = model_cnn.cuda()
                # model_cnn = nn.DataParallel(model_cnn)
                #
                # model_path = os.path.join(opt.auto_start_from_best, opt.id + '_model_best.pth')
                # model.load_state_dict(torch.load(model_path))
                # model = model.cuda()

                del model_cnn
                del model

                torch.cuda.empty_cache()

                model_cnn = models.setup_cnn(opt)
                model_cnn = model_cnn.cuda()
                model_cnn = nn.DataParallel(model_cnn)

                model = models.setup(opt)
                model = model.cuda()

                model_cnn.train()
                model.train()

                update_lr_flag = True


        # start train

        # Update the iteration and epoch
        iteration += 1

        if update_lr_flag:
            if opt.finetune_cnn_after >= 0 and epoch >= opt.finetune_cnn_after:
                finetune_cnn_start = True
            else:
                finetune_cnn_start = False

            optimizer_cnn = train_utils.get_cnn_optimizer(model_cnn, optimizer_cnn, finetune_cnn_start, opt)

            train_utils.update_lr(epoch, optimizer, optimizer_cnn, finetune_cnn_start, opt)

            update_lr_flag = False

        if opt.reinforce_start >= 0 and epoch >= opt.reinforce_start:
            use_reinforce = True
        else:
            use_reinforce = False

        optimizer = get_optimizer(optimizer, epoch, model, model_cnn)

        start_total = time.time()
        start = time.time()

        optimizer.zero_grad()
        if finetune_cnn_start:
            optimizer_cnn.zero_grad()

        # batch data
        data = loader.get_batch('train', batch_size)

        images = data['images']
        bus = None
        if models.has_bu(opt.caption_model):
            bus = data['bus']

        if opt.verbose:
            print('data {:.3f}'.format(time.time() - start))

        start = time.time()

        fc_feats, att_feats, bu_feats = train_cnn(model_cnn, images, bus, fc_expander, att_expander, bu_expander, use_reinforce)

        if opt.verbose:
            print('model_cnn {:.3f}'.format(time.time() - start))

        # get input data
        params['fc_feats'] = fc_feats
        params['att_feats'] = att_feats
        params['bu_feats'] = bu_feats

        # get target data
        params['labels'] = data['labels']
        params['masks'] = data['masks']
        params['tokens'] = data['tokens']
        params['gts'] = data['gts']
        params['targets'] = data['targets']

        # crit_pg, crit_rl, crit_ctc, crit_c, crit_ac, crit,
        train_loss, reward_mean, use_reinforce = train_model(params, iteration, epoch, board)

        # update the gradient
        update_gradient(optimizer, optimizer_cnn, finetune_cnn_start)

        time_batch = time.time() - start_total
        left_time = (opt.save_checkpoint_every - iteration % opt.save_checkpoint_every) * time_batch
        s_left_time = utils.format_time(left_time)
        msg = "id {} iter {} (epoch {}), train_loss = {:.3f}, lr = {} lr_cnn = {} f_cnn = {} rf = {} r = {:.3f} early_stop_cnt = {} time/batch = {:.3f}s time/eval = {}" \
            .format(opt.id, iteration, epoch, train_loss, opt.current_lr, opt.current_cnn_lr, finetune_cnn_start,
                    use_reinforce, reward_mean, early_stop_cnt, time_batch, s_left_time)
        logger.info(msg)

        if opt.use_tensorboard:
            if iteration % opt.tensorboard_for_train_every == 0:
                board.loss_train(train_loss, iteration)

        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if iteration % opt.losses_log_every == 0:
            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            finetune_cnn_history[iteration] = finetune_cnn_start

        # update infos
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['iterators'] = loader.iterators
        infos['best_val_score'] = best_val_score
        infos['opt'] = opt
        infos['val_result_history'] = val_result_history
        infos['loss_history'] = loss_history
        infos['lr_history'] = lr_history
        infos['finetune_cnn_history'] = finetune_cnn_history
        if opt.use_auto_learning_rate == 1:
                infos['train_process_index'] = train_process_index

        if opt.save_snapshot_every > 0 and iteration % opt.save_snapshot_every == 0:
            train_utils.save_model(model, model_cnn, infos, opt)


    loader.terminate()

opt = opts.parse_opt()
logger.init_logger(opt)
train(opt)
