from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import os
import time

import torch.optim as optim

import eval_utils
import misc.Criterion as Criterion
import models
import opts
from data.DataLoader import *


def train(opt):
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.caption_model = "SCST"

    infos = {}
    if opt.start_from is not None and len(opt.start_from) > 0:
        print("start from %s" % (opt.start_from))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[
                    checkme], "Command line argument and saved model disagree on '%s' " % checkme

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model_cnn = models.setup_cnn(opt)
    model_cnn.cuda()

    model = models.setup(opt)
    model.cuda()

    model_cnn.train()
    model.train()

    fc_expander = utils.FeatExpander(5)
    att_expander = utils.FeatExpander(5)

    crit = Criterion.LanguageModelCriterion()
    crit_reinforce = Criterion.SCSTCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)

    # Load the optimizer
    if opt.start_from is not None and len(opt.start_from) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        optimizer_cnn.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer_cnn.pth')))

    finetune_cnn_start = False
    use_reinforce = False
    update_lr_flag = True

    while True:

        if update_lr_flag:
            # Assign the learning rate
            if opt.learning_rate_decay_start >= 0 and epoch >= opt.learning_rate_decay_start:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate

            if opt.finetune_cnn_after >= 0 and epoch >= opt.finetune_cnn_after:
                for p in model_cnn.parameters():
                    p.requires_grad = True
                model_cnn.train()
                finetune_cnn_start = True
            else:
                for p in model_cnn.parameters():
                    p.requires_grad = False
                model_cnn.eval()
                finetune_cnn_start = False

            if opt.cnn_learning_rate_decay_start >= 0 and epoch >= opt.cnn_learning_rate_decay_start:
                frac = (epoch - opt.cnn_learning_rate_decay_start) // opt.cnn_learning_rate_decay_every
                decay_factor = opt.cnn_learning_rate_decay_rate ** frac
                opt.current_cnn_lr = opt.cnn_learning_rate * decay_factor
                utils.set_lr(optimizer_cnn, opt.current_cnn_lr)  # set the decayed rate
            else:
                opt.current_cnn_lr = opt.cnn_learning_rate

            update_lr_flag = False


        start_total = time.time()

        data = loader.get_batch('train')
        vocab = loader.get_vocab()

        images = torch.from_numpy(data['images']).cuda()
        images = utils.prepro(images, False)
        images = Variable(images, requires_grad=False)

        labels = torch.from_numpy(data['labels']).cuda()
        labels = Variable(labels, requires_grad=False)

        fc_feats, att_feats = model_cnn(images)

        fc_feats_ext = fc_expander(fc_feats)
        att_feats_ext = att_expander(att_feats)

        optimizer.zero_grad()
        if opt.finetune_cnn_after >= 0 and epoch >= opt.finetune_cnn_after:
            optimizer_cnn.zero_grad()

        reward = 0
        reward1 = 0
        if opt.reinforce_start >= 0 and epoch >= opt.reinforce_start:
            use_reinforce = True
            output1, seq1 = model(fc_feats_ext, att_feats_ext, labels, "test")
            output, seq = model(fc_feats_ext, att_feats_ext, labels, "train")
            loss, reward, reward1 = crit_reinforce(output, output1, seq, seq1, labels, vocab)
            loss.backward()
        else:
            use_reinforce = False
            output, _ = model(fc_feats_ext, att_feats_ext, labels, "xent")
            loss = crit(output, labels)
            loss.backward()


        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if opt.finetune_cnn_after >= 0 and epoch >= opt.finetune_cnn_after:
            utils.clip_gradient(optimizer_cnn, opt.grad_clip)
            optimizer_cnn.step()

        train_loss = loss.data[0]

        print("iter {} (epoch {}), train_loss = {:.3f}, lr = {} lr_cnn = {} finetune_cnn = {} use_reinforce = {} reward = {} reward1 = {} time/batch = {:.3f}" \
              .format(iteration, epoch, train_loss, opt.current_lr, opt.current_cnn_lr, finetune_cnn_start, use_reinforce, reward, reward1, time.time() - start_total))

        # Update the iteration and epoch
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
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json,
                           'caption_model': 'SCST'}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats, str_stats = eval_utils.eval_split(model_cnn, model, crit, loader, eval_kwargs)

            if not os.path.exists(opt.eval_result_path):
                os.makedirs(opt.eval_result_path)

            eval_result_file = os.path.join(opt.eval_result_path, opt.id + ".csv")
            with open(eval_result_file, 'a') as f:
                f.write(str_stats + "\n")

            predictions_file = os.path.join(opt.eval_result_path, opt.id + ".json")
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f)

            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True:  # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                if not os.path.exists(opt.checkpoint_path):
                    os.makedirs(opt.checkpoint_path)

                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))

                checkpoint_path_cnn = os.path.join(opt.checkpoint_path, 'model_cnn.pth')
                torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
                print("model cnn saved to {}".format(checkpoint_path_cnn))

                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                print("optimizer saved to {}".format(optimizer_path))

                optimizer_path_cnn = os.path.join(opt.checkpoint_path, 'optimizer_cnn.pth')
                torch.save(optimizer_cnn.state_dict(), optimizer_path_cnn)
                print("optimizer cnn saved to {}".format(optimizer_path_cnn))

                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['val_result_history'] = val_result_history
                infos['loss_history'] = loss_history
                infos['lr_history'] = lr_history
                infos['vocab'] = loader.get_vocab()

                info_path = os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl')
                with open(info_path, 'wb') as f:
                    cPickle.dump(infos, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model_best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))

                    checkpoint_path_cnn = os.path.join(opt.checkpoint_path, 'model_cnn_best.pth')
                    torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
                    print("model cnn saved to {}".format(checkpoint_path_cnn))

                    info_path = os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '_best.pkl')
                    with open(info_path, 'wb') as f:
                        cPickle.dump(infos, f)

        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
train(opt)