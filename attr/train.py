from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import time
import cPickle
import os
import torch.nn as nn
import attr.models as models
import argparse
import misc.utils
import tensorboard
from attr.DataLoaderThread import *


def train(opt):

    if opt.use_tensorboard:
        board = tensorboard.TensorBoard()
        board.start(opt.id)

    loader = DataLoaderThread(opt)

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    vocab = loader.get_vocab()

    infos = {}
    if opt.start_from_best is not None and len(opt.start_from_best) > 0:
        print("start best from %s" % (opt.start_from_best))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from_best, 'infos_' + opt.id + '_best.pkl')) as f:
            infos = cPickle.load(f)
    elif opt.start_from is not None and len(opt.start_from) > 0:
        print("start from %s" % (opt.start_from))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')) as f:
            infos = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model_cnn = models.setup_cnn(opt)
    model_cnn = nn.DataParallel(model_cnn).cuda()

    update_lr_flag = True

    model_cnn.train()

    crit = nn.BCELoss()

    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=opt.cnn_learning_rate, betas=(opt.cnn_optim_alpha, opt.cnn_optim_beta),
                               eps=opt.cnn_optim_epsilon, weight_decay=opt.cnn_weight_decay)
    while True:

        if update_lr_flag:
            # Assign the learning rate
            if opt.cnn_learning_rate_decay_start >= 0 and epoch >= opt.cnn_learning_rate_decay_start:
                frac = (epoch - opt.cnn_learning_rate_decay_start) // opt.cnn_learning_rate_decay_every
                decay_factor = opt.cnn_learning_rate_decay_rate ** frac
                opt.current_cnn_lr = opt.cnn_learning_rate * decay_factor
                misc.utils.set_lr(optimizer_cnn, opt.current_cnn_lr)  # set the decayed rate
            else:
                opt.current_cnn_lr = opt.cnn_learning_rate

            update_lr_flag = False


        start_total = time.time()

        start = time.time()

        # batch data
        data = loader.get_batch('train')

        images = data['images']
        labels = data['labels']

        if opt.verbose:
            print('data {:.3f}'.format(time.time() - start))

        optimizer_cnn.zero_grad()

        # forward
        start = time.time()

        output = model_cnn(images)

        if opt.verbose:
            print('model {:.3f}'.format(time.time() - start))

        # compute the loss
        start = time.time()

        loss = crit(output, labels)
        if opt.verbose:
            print('crit {:.3f}'.format(time.time() - start))

        # backward
        start = time.time()
        loss.backward()
        if opt.verbose:
            print('loss {:.3f}'.format(time.time() - start))

        # show information
        train_loss = loss.data[0]

        # update the gradient
        misc.utils.clip_gradient(optimizer_cnn, opt.grad_clip)
        optimizer_cnn.step()


        print("iter {} (epoch {}), train_loss = {:.3f}, lr_cnn = {} time/batch = {:.3f}" \
              .format(iteration, epoch, train_loss, opt.current_cnn_lr, time.time() - start_total))

        if opt.use_tensorboard:
            board.loss_train(train_loss, iteration)

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_cnn_lr

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': opt.val_split,
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))

            print("start eval ...")

            val_loss, accuracy = eval(model_cnn, crit, loader, eval_kwargs)

            if opt.use_tensorboard:
                board.accuracy({'accuracy':accuracy}, iteration)
                board.loss_val(val_loss, iteration)

            print("end eval ...")

            val_result_history[iteration] = {'loss': val_loss, 'accuracy': accuracy}

            print('accuracy', accuracy)

            # Save model if is improving on validation result
            current_score = accuracy

            best_flag = False
            if True:  # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                if not os.path.exists(opt.checkpoint_path):
                    os.makedirs(opt.checkpoint_path)

                checkpoint_path_cnn = os.path.join(opt.checkpoint_path, 'model_cnn_' + opt.id +  '.pth')
                torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
                print("model cnn saved to {}".format(checkpoint_path_cnn))

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
                    if not os.path.exists(opt.checkpoint_best_path):
                        os.makedirs(opt.checkpoint_best_path)

                    checkpoint_path_cnn = os.path.join(opt.checkpoint_best_path, 'model_cnn_' + opt.id + '_best.pth')
                    torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
                    print("model cnn saved to {}".format(checkpoint_path_cnn))

                    info_path = os.path.join(opt.checkpoint_best_path, 'infos_' + opt.id + '_best.pkl')
                    with open(info_path, 'wb') as f:
                        cPickle.dump(infos, f)

        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

def get_accuracy(input, label):
    input = input.view(-1)
    label = label.view(-1)
    total = len(label)
    correct = ((input==1) == (label > 0.2)).sum().float()
    accuracy = correct / total
    return accuracy

def get_anno(input, vocab):
    print(input.size())
    preditions = []
    for i in xrange(input.size(0)):
        predition = []
        for j in xrange(input.size(1)):
            if input[i, j].data[0] == 1:
                predition.append(vocab[str(j+1)])
        preditions.append(predition)
    return preditions

def eval(model_cnn, crit, loader, eval_kwargs):
    verbose_eval = eval_kwargs.get('verbose_eval', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    coco_caption_path = eval_kwargs.get('coco_caption_path', 'coco-caption')
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 2)

    # Make sure in the evaluation mode
    model_cnn.eval()

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0

    vocab = loader.get_vocab()
    vocab_size = loader.get_vocab_size()
    count = 0
    total_acc = 0
    while True:
        data = loader.get_batch(split)
        n = n + batch_size

        images = data['images']
        labels = data['labels']

        print('labels', get_anno(labels, vocab))

        outputs = model_cnn(images)

        print('outputs', get_anno(outputs>0.2, vocab))


        loss = crit(outputs, labels)
        loss.backward()

        loss_sum = loss_sum + loss.data[0]
        loss_evals = loss_evals + 1

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        acc = get_accuracy(outputs, labels)

        total_acc += acc
        count += 1

        if verbose_eval:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss.data[0]))

        if data['bounds']['wrapped']:
            break
        if n >= val_images_use:
            break

    avg_acc = total_acc / count

    # Switch back to training mode
    model_cnn.train()

    return loss_sum / loss_evals, avg_acc.data[0]



parser = argparse.ArgumentParser()

parser.add_argument('--input_json', type=str, default='/media/amds/data3/dataset/mscoco/data_attr.json',
                    help='path to the json file containing additional info and vocab')
parser.add_argument('--input_h5', type=str, default='/media/amds/data3/dataset/mscoco/data_attr.h5',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_cnn_resnet152', type=str,
                    default='/media/amds/data2/dataset/resnet/resnet152-b121ed2d.pth',
                    help='path to the cnn')
parser.add_argument('--input_cnn_resnet200', type=str, default='/media/amds/data2/dataset/resnet/resnet_200_cpu.pth',
                    help='path to the cnn')
parser.add_argument('--input_cnn_resnext_101_32x4d', type=str,
                    default='/media/amds/data2/dataset/resnet/resnext_101_32x4d.pth',
                    help='path to the cnn')
parser.add_argument('--input_cnn_resnext_101_64x4d', type=str,
                    default='/media/amds/data2/dataset/resnet/resnext_101_64x4d.pth',
                    help='path to the cnn')
# tensorboard
parser.add_argument('--use_tensorboard', type=bool, default=False,
                    help='use tensorboard to show the log')

parser.add_argument('--start_from', type=str, default='',
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                    'infos.pkl'         : configuration;
                    'checkpoint'        : paths to model file(s) (created by tf).
                                          Note: this file contains absolute paths, be careful when moving files around;
                    'model.ckpt-*'      : file(s) with model definition (created by tf)
                """)
parser.add_argument('--start_from_best', type=str, default='',
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
parser.add_argument('--verbose', type=str, default=False,
                    help='print the time')

parser.add_argument('--cnn_model', type=str, default="resnext_101_32x4d",
                    help='resnet_152, resnet_152_rnn, resnet_200, resnext_101_32x4d, resnext_101_64x4d')

parser.add_argument('--image_size', type=int, default=256,
                    help='size of the rnn in number of hidden nodes in each layer')

# Optimization: General
parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='minibatch size')
parser.add_argument('--grad_clip', type=float, default=0.1,  # 5.,
                    help='clip gradients at this value')
parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
parser.add_argument('--finetune_cnn_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
parser.add_argument('--finetune_cnn_type', type=int, default=0,
                    help='0 all, 1 part')
parser.add_argument('--seq_per_img', type=int, default=1,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

# Optimization: for the CNN
parser.add_argument('--cnn_optim', type=str, default='adam',
                    help='optimization to use for CNN')
parser.add_argument('--cnn_optim_alpha', type=float, default=0.8,
                    help='alpha for momentum of CNN')
parser.add_argument('--cnn_optim_beta', type=float, default=0.999,
                    help='beta for momentum of CNN')
parser.add_argument('--cnn_optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
parser.add_argument('--cnn_learning_rate', type=float, default=1e-5,
                    help='learning rate for the CNN')
parser.add_argument('--cnn_weight_decay', type=float, default=0,
                    help='L2 weight decay just for the CNN')

parser.add_argument('--cnn_learning_rate_decay_start', type=int, default=0,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
parser.add_argument('--cnn_learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--cnn_learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')

# Evaluation/Checkpointing
parser.add_argument('--val_split', type=str, default='test',
                    help='use to valid result')
parser.add_argument('--val_images_use', type=int, default=2,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
parser.add_argument('--save_checkpoint_every', type=int, default=10,
                    help='how often to save a model checkpoint (in iterations)?')
parser.add_argument('--checkpoint_path', type=str, default='../caption_checkpoint/resnet152/',
                    help='directory to store checkpointed models')
parser.add_argument('--checkpoint_best_path', type=str, default='../caption_checkpoint_best/resnet152/',
                    help='directory to store checkpointed models')
parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')
parser.add_argument('--data_norm', type=bool, default=True,
                    help='control the stop weight of loss')

# misc
parser.add_argument('--id', type=str, default='dasc_attr',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')


opt = parser.parse_args()
train(opt)