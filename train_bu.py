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

def save_model_conf(model, opt):

    model_file = os.path.join(opt.eval_result_path, opt.id + '_model.txt')
    with open(model_file, 'w') as f:
        f.write("{}\n".format(model))

def train_reinforce(params):

    model = params['model']
    bu_feats = params['bu_feats']
    labels = params['labels']
    masks = params['masks']
    vocab = params['vocab']
    crit_rl = params['crit_rl']
    gts = params['gts']

    # forward
    start = time.time()

    sample_seq, sample_seqLogprobs = model.sample(bu_feats, {'sample_max': 0})
    greedy_seq, greedy_seqLogprobs = model.sample(bu_feats, {'sample_max': 1})

    if opt.verbose:
        print('model {:.3f}'.format(time.time() - start))

    # compute the loss
    start = time.time()
    # seq, seqLogprobs, seq1, target, vocab
    loss, reward_mean, sample_mean, greedy_mean = crit_rl(sample_seq, sample_seqLogprobs, greedy_seq, gts)
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


def train_normal(params):

    model = params['model']
    bu_feats = params['bu_feats']
    labels = params['labels']
    masks = params['masks']
    crit = params['crit']

    # forward
    start = time.time()

    output = model(bu_feats, labels)

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


def save_result(str_stats, predictions):
    eval_result_file = os.path.join(opt.eval_result_path, opt.id + ".csv")
    with open(eval_result_file, 'a') as f:
        f.write(str_stats + "\n")

    predictions_file = os.path.join(opt.eval_result_path, opt.id + ".json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

def save_model(model, infos):

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    checkpoint_path = os.path.join(opt.checkpoint_path, opt.id + '_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    info_path = os.path.join(opt.checkpoint_path, opt.id + '_infos.pkl')
    with open(info_path, 'wb') as f:
        cPickle.dump(infos, f)


def save_model_best(model, infos):
    if not os.path.exists(opt.checkpoint_best_path):
        os.makedirs(opt.checkpoint_best_path)

    checkpoint_path = os.path.join(opt.checkpoint_best_path, opt.id + '_model_best.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    info_path = os.path.join(opt.checkpoint_best_path, opt.id + '_infos_best.pkl')
    with open(info_path, 'wb') as f:
        cPickle.dump(infos, f)

def update_lr(epoch, optimizer):
    # Assign the learning rate
    if opt.learning_rate_decay_start >= 0 and epoch >= opt.learning_rate_decay_start:
        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
        utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
    else:
        opt.current_lr = opt.learning_rate

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
        board.start(opt.id)

        # board = trans_client.TransClient()
        # board.start(opt.id)

    print(opt.cnn_model)

    loader = DataLoaderThreadBu(opt)

    # if models.is_inception(opt.cnn_model):
    #     loader = DataLoaderThreadNew(opt)
    # else:
    #     loader = DataLoaderThread(opt)

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    vocab = loader.get_vocab()
    batch_size = loader.batch_size

    try:
        infos = load_infos(opt)
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

    model = models.setup(opt)
    model.cuda()

    save_model_conf(model, opt)

    update_lr_flag = True

    model.train()

    if opt.seq_per_img > 1:
        bu_expander = utils.FeatExpander(opt.seq_per_img)

    crit = Criterion.LanguageModelWeightCriterion()
    crit_rl = None

    # print(model_cnn)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(opt.optim_alpha, opt.optim_beta), eps=opt.optim_epsilon)

    early_stop_cnt = 0

    params = {}
    params['model'] = model
    params['crit'] = crit
    params['vocab'] = vocab

    while True:

        try:
            if update_lr_flag:
                update_lr(epoch, optimizer)
                update_lr_flag = False

            start_total = time.time()
            start = time.time()

            optimizer.zero_grad()

            # batch data
            data = loader.get_batch('train', batch_size)

            labels = data['labels']
            masks = data['masks']
            tokens = data['tokens']
            gts = data['gts']
            bu_feats = data['bus']

            if opt.verbose:
                print('data {:.3f}'.format(time.time() - start))

            params['labels'] = labels
            params['masks'] = masks
            params['tokens'] = tokens
            params['gts'] = gts

            if opt.seq_per_img > 1:
                bu_feats = bu_expander(bu_feats)
            params['bu_feats'] = bu_feats

            if opt.reinforce_start >= 0 and epoch >= opt.reinforce_start:
                use_reinforce = True
                if crit_rl is None:
                    if opt.is_aic_data:
                        crit_rl = Criterion.RewardCriterionAIC(opt, vocab)
                    else:
                        crit_rl = Criterion.RewardCriterion(opt)

                params['crit_rl'] = crit_rl

                train_loss, reward_mean, sample_mean, greedy_mean = train_reinforce(params)

                if opt.use_tensorboard:
                    board.val("sample_mean", sample_mean, iteration)
                    board.val("greedy_mean", greedy_mean, iteration)
            else:
                use_reinforce = False
                train_loss, reward_mean = train_normal(params)

            # update the gradient
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()

            msg = "iter {} (epoch {}), train_loss = {:.3f}, lr = {} rf = {} r = {:.3f} time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, opt.current_lr,
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

                    val_loss, predictions, lang_stats, str_stats = eval_utils.eval_split_without_cnn(model, crit, loader, eval_kwargs)

                    if opt.use_tensorboard:
                        board.accuracy(lang_stats, iteration)
                        board.loss_val(val_loss, iteration)

                    print("end eval ...")

                    msg = "iteration = {} val_loss = {} str_stats = {}".format(iteration, val_loss, str_stats)
                    notifier.send(opt.id + " val result", opt.id + " :\n" + msg)
                    logger.info(msg)

                    save_result(str_stats + ',' + str(val_loss), predictions)

                    val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                     'predictions': predictions}

                    # Save model if is improving on validation result
                    if opt.language_eval == 1:
                        eval_metric = opt.eval_metric
                        current_score = lang_stats[eval_metric]
                    else:
                        current_score = -val_loss

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
                save_model(model, infos)

                if best_flag:
                    save_model_best(model, infos)
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                    # if early_stop_cnt > 10:
                    #     msg = 'early stop'
                    #     notifier.send(opt.id + " " + msg, opt.id + " :\n" + msg)
                    #     logger.info(msg)
                    #     break

            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

            if opt.is_eval_start:
                iteration += 1
        # except Exception, e:
        #     notifier.send(opt.id + " training exception", e.message)
        #     print(e.message)
        finally:
            pass

    loader.terminate()

opt = opts.parse_opt()
logger.init_logger(opt)
train(opt)