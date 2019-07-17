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
        board = tensorboard.TensorBoard()
        board.start(opt.id, opt.tensorboard_ip)

        # board = trans_client.TransClient()
        # board.start(opt.id)

    print(opt.cnn_model)

    loader = DataLoaderThreadBu(opt)

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    vocab = loader.get_vocab()
    batch_size = loader.batch_size

    try:
        if opt.is_load_infos == 1:
            infos = train_utils.load_infos(opt)
        else:
            infos = {}
    except:
        infos = {}
        print('load infos error')


    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score = None

    model_cnn = models.setup_cnn(opt)
    model_cnn = nn.DataParallel(model_cnn.cuda())

    model = models.setup(opt)
    model.cuda()

    train_utils.save_model_conf(model_cnn, model, opt)

    update_lr_flag = True

    model_cnn.train()
    model.train()

    if opt.seq_per_img > 1:
        fc_expander = utils.FeatExpander(opt.seq_per_img)
        att_expander = utils.FeatExpander(opt.seq_per_img)
        bu_expander = utils.FeatExpander(opt.seq_per_img)

    # crit = Criterion.LanguageModelWeightNewCriterion()
    crit = Criterion.LanguageModelWithProbWeightCriterion(opt.prob_weight_alpha)

    crit_rl = None

    # print(model_cnn)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(opt.optim_alpha, opt.optim_beta), eps=opt.optim_epsilon)
    optimizer_cnn = None
    finetune_cnn_start = False

    early_stop_cnt = 0

    params = {}
    params['model'] = model
    params['crit'] = crit
    params['vocab'] = vocab

    while True:

        # try:
        if update_lr_flag:
            if opt.finetune_cnn_after >= 0 and epoch >= opt.finetune_cnn_after:
                finetune_cnn_start = True
            else:
                finetune_cnn_start = False

            optimizer_cnn = train_utils.finetune_cnn(model_cnn, optimizer_cnn, finetune_cnn_start, opt)

            train_utils.update_lr(epoch, optimizer, optimizer_cnn, finetune_cnn_start, opt)

            update_lr_flag = False

        start_total = time.time()
        start = time.time()

        optimizer.zero_grad()
        if finetune_cnn_start:
            optimizer_cnn.zero_grad()

        # batch data
        data = loader.get_batch('train', batch_size)

        images = data['images']
        labels = data['labels']
        masks = data['masks']
        tokens = data['tokens']
        gts = data['gts']

        if opt.verbose:
            print('data {:.3f}'.format(time.time() - start))

        # train cnn
        fc_feats, att_feats, bu_feats = model_cnn(images)
        if opt.seq_per_img > 1:
            fc_feats = fc_expander(fc_feats)
            att_feats = att_expander(att_feats)
            bu_feats = bu_expander(bu_feats)


        params['fc_feats'] = fc_feats
        params['att_feats'] = att_feats
        params['bu_feats'] = bu_feats
        params['labels'] = labels
        params['masks'] = masks
        params['tokens'] = tokens
        params['gts'] = gts

        if opt.reinforce_start >= 0 and epoch >= opt.reinforce_start:
            use_reinforce = True

            if crit_rl is None:
                if opt.is_aic_data:
                    crit_rl = Criterion.RewardCriterionAIC(opt, vocab)
                else:
                    crit_rl = Criterion.RewardCriterion(opt)

            params['crit_rl'] = crit_rl

            train_loss, reward_mean, sample_mean, greedy_mean = train_utils.train_reinforce(params, opt)

            if opt.use_tensorboard:
                board.val("sample_mean", sample_mean, iteration)
                board.val("greedy_mean", greedy_mean, iteration)

        else:
            use_reinforce = False
            params['crit'] = crit

            train_loss, reward_mean = train_utils.train_with_prob_weight(params, opt)

        # update the gradient
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if finetune_cnn_start:
            utils.clip_gradient(optimizer_cnn, opt.grad_clip)
            optimizer_cnn.step()

        msg = "iter {} (epoch {}), train_loss = {:.3f}, lr = {} lr_cnn = {} f_cnn = {} rf = {} r = {:.3f} time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, opt.current_lr, opt.current_cnn_lr, finetune_cnn_start,
                    use_reinforce, reward_mean, time.time() - start_total)
        logger.info(msg)

        if opt.use_tensorboard:
            board.loss_train(train_loss, iteration)

        # Update the iteration and epoch
        if not opt.is_eval_start:
            iteration += 1

        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):

            if opt.is_every_eval:

                # eval model
                eval_kwargs = {'split': opt.val_split,
                               'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))

                print("start eval ...")

                val_loss, predictions, lang_stats, str_stats = eval_utils.eval_split_with_region_bu(model_cnn, model, crit, loader,
                                                                                     eval_kwargs)

                if opt.use_tensorboard:
                    board.accuracy(lang_stats, iteration)
                    board.loss_val(val_loss, iteration)


                print("end eval ...")

                msg = "iteration = {} val_loss = {} str_stats = {}".format(iteration, val_loss, str_stats)
                notifier.send(opt.id + " val result", opt.id + " :\n" + msg)
                logger.info(msg)

                train_utils.save_result(str_stats + ',' + str(val_loss), predictions, opt)

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

            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            infos['loss_history'] = loss_history
            infos['lr_history'] = lr_history
            infos['vocab'] = loader.get_vocab()

            train_utils.save_model(model, model_cnn, infos, opt)

            if best_flag:
                train_utils.save_model_best(model, model_cnn, infos, opt)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

        if opt.is_eval_start:
            iteration += 1

    loader.terminate()

opt = opts.parse_opt()
logger.init_logger(opt)
train(opt)