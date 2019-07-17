from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle

import torch
from torch.autograd import Variable

import misc.utils as utils
import models
import zipfile
from data.DataLoaderRaw import *
import time


# CUDA_VISIBLE_DEVICES=0 python eval_aic.py --start 0 --num 7500
# CUDA_VISIBLE_DEVICES=1 python eval_aic.py --start 7500 --num 7500
# CUDA_VISIBLE_DEVICES=2 python eval_aic.py --start 15000 --num 7500
# CUDA_VISIBLE_DEVICES=3 python eval_aic.py --start 22500 --num 7500

# CUDA_VISIBLE_DEVICES=0 python eval_aic.py --id aic_weight_up_1 --start 0 --num 15000
# CUDA_VISIBLE_DEVICES=1 python eval_aic.py --id aic_weight_up_1 --start 15000 --num 15000
# CUDA_VISIBLE_DEVICES=2 python eval_aic.py --id aic_weight_up_2 --start 0 --num 15000
# CUDA_VISIBLE_DEVICES=3 python eval_aic.py --id aic_weight_up_2 --start 15000 --num 15000

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    # Input arguments and options
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--batch_size', type=int, default=14,
                    help='if > 0 then overrule, otherwise load from checkpoint.')

    # Sampling options
    parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

    # For evaluation on a folder of images:
    parser.add_argument('--datasets', type=list, default=['caption_test1_images_20170923'],
                        help='caption_test1_images_20170923')
    parser.add_argument('--image_folder', type=str, default='/home/amds/caption/dataset/aic/ai_challenger_caption_test1_20170923',
                    help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--start_from_best', type=str, default='/home/amds/caption/caption_checkpoint_best/aic2',
                        help="")
    parser.add_argument('--output_dir', type=str, default='/home/amds/caption/caption_aic',
                        help='')
    parser.add_argument('--id', type=str, default='aic_weight_up_1',
                    help='')

    parser.add_argument('--start', type=int, default=0,
                        help='')
    parser.add_argument('--num', type=int, default=7500,
                        help='')

    # bottom up attention
    parser.add_argument('--input_bu', type=str, default='/home/amds/data/code/bottom-up-attention/output/aic_test_resnet101_fasterRcnn_lmdb',
                        help='')
    parser.add_argument('--bu_size', type=int, default=36,
                        help='use bottom up attention')
    parser.add_argument('--bu_feat_size', type=int, default=2048,
                        help='use bottom up attention')


    return parser.parse_args()


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


def eval_split(model_cnn, model, loader, eval_kwargs={}):

    verbose_eval = eval_kwargs.get('verbose_eval', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    caption_model = eval_kwargs.get('caption_model', '')
    batch_size = eval_kwargs.get('batch_size', 1)
    use_bu_att = eval_kwargs.get('use_bu_att', False)

    split = ''
    loader.reset_iterator(split)
    n = 0
    predictions = []
    vocab = loader.get_vocab()

    while True:

        start = time.time()

        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        images = torch.from_numpy(data['images']).cuda()
        images = utils.prepro_norm(images, False)
        images = Variable(images, requires_grad=False)

        if models.is_only_fc_feat(caption_model):
            fc_feats = model_cnn(images)
        else:
            fc_feats, att_feats = model_cnn(images)

        if models.has_bu(caption_model):
            bu_feats = torch.from_numpy(data['bus']).cuda()
            bu_feats = Variable(bu_feats, requires_grad=False)

        if models.is_only_fc_feat(caption_model):
            output = model.sample(fc_feats, {'beam_size': beam_size})
        elif models.has_bu(caption_model):
            output = model.sample(fc_feats, att_feats, bu_feats, {'beam_size': beam_size})
        else:
            output = model.sample(fc_feats, att_feats, {'beam_size': beam_size})

        seq = output[0]

        # sents
        sents = utils.decode_sequence_aic(vocab, seq)
        for k, sent in enumerate(sents):
            image_id = data['infos'][k]['id']
            # print(image_id, sent)
            # image_id = int(image_id.split('_')[2])
            entry = {'image_id': image_id, 'caption': sent}
            predictions.append(entry)
            if verbose_eval:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        for i in range(n - ix1):
            predictions.pop()
        if verbose_eval:
            span_time = time.time()-start
            left_time = (ix1-ix0)*span_time/batch_size
            s_left_time = format_time(left_time)
            print('evaluating validation preformance... %d/%d %.3fs left:%s' % (ix0, ix1, span_time, s_left_time))

        if data['bounds']['wrapped']:
            break

    return predictions

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
        with open(os.path.join(opt.start_from, opt.id + '_infos_.pkl')) as f:
            infos = cPickle.load(f)
    return infos

def main():

    opt = parse_args()

    #
    opt.output_dir = os.path.join(opt.output_dir, opt.id)

    # make dirs
    print(opt.output_dir)
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    # Load infos
    infos = load_infos(opt)

    ignore = ["id", "batch_size", "beam_size", "start_from_best", "input_bu"]
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

    str_id = ''.join(opt.id.split('_'))

    # path_zip = opt.output_dir + '/results.zip'

    # zipf = zipfile.ZipFile(path_zip, 'w', zipfile.ZIP_DEFLATED)

    use_bu_att = models.has_bu(opt.caption_model)
    print(use_bu_att)

    for dataset in opt.datasets:

        loader = DataLoaderRaw({'folder_path': os.path.join(opt.image_folder, dataset),
                                'batch_size': opt.batch_size,
                                'start': opt.start,
                                'num': opt.num,
                                'use_bu_att': use_bu_att,
                                'input_bu': opt.input_bu,
                                'bu_size': opt.bu_size,
                                'bu_feat_size': opt.bu_feat_size})
        loader.ix_to_word = infos['vocab']

        # Set sample options
        predictions = eval_split(model_cnn, model, loader,  vars(opt))

        path_json = opt.output_dir + '/captions_' + dataset + '_' + str_id + '_' + str(opt.start) + '_results.json'

        json.dump(predictions, open(path_json, 'w'))

        # zipf.write(path_json)

    # zipf.close()

if __name__ == '__main__':
    main()