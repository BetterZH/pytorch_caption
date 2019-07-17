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


def parse_args():
    # Input arguments and options
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--batch_size', type=int, default=1,
                    help='if > 0 then overrule, otherwise load from checkpoint.')

    # Sampling options
    parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

    # For evaluation on a folder of images:
    parser.add_argument('--image_folder', type=str, default='/media/amds/data4/dataset/test',
                    help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--start_from_best', type=str, default='/media/amds/data4/checkpoint/caption_checkpoint_best/lstm',
                        help="")
    parser.add_argument('--output_dir', type=str, default='/media/amds/data4/caption_result/test',
                        help='')
    # misc
    parser.add_argument('--id', type=str, default='dasc_np',
                    help='')

    parser.add_argument('--checkpoint_best_path', type=str, default='/media/amds/data4/checkpoint/caption_checkpoint_best/test',
                        help="")

    return parser.parse_args()

def save_model_best(model, model_cnn, infos, opt):
    if not os.path.exists(opt.checkpoint_best_path):
        os.makedirs(opt.checkpoint_best_path)

    checkpoint_path = os.path.join(opt.checkpoint_best_path, 'model_' + opt.id + '_best.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    checkpoint_path_cnn = os.path.join(opt.checkpoint_best_path, 'model_cnn_' + opt.id + '_best.pth')
    torch.save(model_cnn.state_dict(), checkpoint_path_cnn)
    print("model cnn saved to {}".format(checkpoint_path_cnn))

    info_path = os.path.join(opt.checkpoint_best_path, 'infos_' + opt.id + '_best.pkl')
    with open(info_path, 'wb') as f:
        cPickle.dump(infos, f)

def eval_split(model_cnn, model, loader, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 1)
    output_dir = eval_kwargs.get('output_dir', '')

    split = ''
    loader.reset_iterator(split)
    n = 0
    predictions = []
    vocab = loader.get_vocab()

    dir_fc = os.path.join(output_dir, 'fc')
    dir_att = os.path.join(output_dir, 'att')

    print(dir_fc)
    print(dir_att)

    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    while True:
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        images = torch.from_numpy(data['images']).cuda()
        images = utils.prepro_norm(images, False)
        images = Variable(images, requires_grad=False)

        if models.is_only_fc_feat(caption_model):
            fc_feats = model_cnn(images)
        else:
            fc_feats, att_feats = model_cnn(images)

        if models.is_only_fc_feat(caption_model):
            seq, _ = model.sample(fc_feats, {'beam_size': beam_size})
        else:
            seq, _ = model.sample(fc_feats, att_feats, {'beam_size': beam_size})

        # sents
        sents = utils.decode_sequence(vocab, seq)

        for k, sent in enumerate(sents):

            # att_batch = y[k].data.cpu().float().numpy()
            # np.savez_compressed(os.path.join(dir_att, data['infos'][k]['id']), x = att_batch)

            # fc_batch = fc_feats[k].data.cpu().float().numpy()
            # att_batch = att_feats[k].data.cpu().float().numpy()
            #
            # np.savez_compressed(os.path.join(dir_fc, data['infos'][k]['id']), x = fc_batch)
            # np.savez_compressed(os.path.join(dir_att, data['infos'][k]['id']), x = att_batch)

            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            print('evaluating validation preformance... %d/%d' % (ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break

    return predictions

def load_infos(opt):
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
    return infos

def main():

    opt = parse_args()

    # make dirs
    print(opt.output_dir)
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    # Load infos
    infos = load_infos(opt)

    ignore = ["id", "batch_size", "beam_size", "start_from_best", "checkpoint_best_path"]
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    print(opt)

    # Setup the model
    model_cnn = models.setup_cnn(opt)
    model_cnn.cuda()

    model = models.setup(opt)
    model.cuda()

    # Make sure in the evaluation mode
    model_cnn.eval()
    model.eval()

    save_model_best(model, model_cnn, infos, opt)

    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'batch_size': opt.batch_size})
    loader.ix_to_word = infos['vocab']

    # Set sample options
    predictions = eval_split(model_cnn, model, loader,  vars(opt))

    json.dump(predictions, open(opt.output_dir + '/result.json', 'w'))

if __name__ == '__main__':
    main()