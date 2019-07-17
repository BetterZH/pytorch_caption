from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle

import torch
from torch.autograd import Variable

import misc.utils as utils
import models
from data.DataLoaderThreadBu import *
from data.DataLoaderThreadNew import *
import train_utils
import eval_utils

def parse_args():
    # Input arguments and options
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--input_json', type=str, default='/media/amds/data/dataset/mscoco/data_coco.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_h5', type=str, default='/media/amds/data/dataset/mscoco/data_coco.h5',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_anno', type=str, default='/media/amds/data/dataset/mscoco/anno_coco.json',
                        help='')
    parser.add_argument('--images_root', default='/media/amds/disk/dataset/mscoco',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--coco_caption_path', type=str, default='/media/amds/data1/code/coco-caption',
                    help='coco_caption_path')
    # Basic options
    parser.add_argument('--eval_split', type=str, default='test',
                        help='eval split')
    parser.add_argument('--batch_size', type=int, default=12,
                    help='if > 0 then overrule, otherwise load from checkpoint.')

    # Sampling options
    parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

    # For evaluation on a folder of images:
    parser.add_argument('--start_from_best', type=str, default='/media/amds/disk/caption_result/dasc_sup',
                        help="")
    parser.add_argument('--eval_result_path', type=str, default='/media/amds/disk/caption_result/server_result',
                        help='')
    parser.add_argument('--id', type=str, default='dasc_sup_1_rl_1',
                    help='')

    #
    parser.add_argument('--transformer_decoder_type', type=str, default='ImageDecoder',
                        help='ImageDecoder,'
                             'ImageAttentionDecoder')

    return parser.parse_args()

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

def save_result(str_stats, predictions, opt):
    eval_result_file = os.path.join(opt.eval_result_path, "eval_" + opt.id + "_beam" + ".csv")
    with open(eval_result_file, 'a') as f:
        f.write(str_stats + "\n")

    predictions_file = os.path.join(opt.eval_result_path, "eval_" + opt.id + "_beam_" + str(opt.beam_size) + ".json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

def main():

    opt = parse_args()

    # make dirs
    print(opt.eval_result_path)
    if not os.path.isdir(opt.eval_result_path):
        os.makedirs(opt.eval_result_path)

    # Load infos
    infos = load_infos(opt)

    ignore = ["id", "input_json", "input_h5", "input_anno",
              "images_root", "coco_caption_path",
              "batch_size", "beam_size",
              "start_from_best", "eval_result_path"]
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    # print(opt)

    # Setup the model
    model_cnn = models.setup_cnn(opt)
    model_cnn.cuda()

    model = models.setup(opt)
    model.cuda()

    # Make sure in the evaluation mode
    model_cnn.eval()
    model.eval()

    if models.has_bu(opt.caption_model) or \
            models.has_sub_regions(opt.caption_model) or \
            models.has_sub_region_bu(opt.caption_model):
        loader = DataLoaderThreadBu(opt)
        print("DataLoaderThreadBu")
    else:
        loader = DataLoaderThreadNew(opt)
        print("DataLoaderThreadNew")

    loader.ix_to_word = infos['vocab']

    eval_kwargs = {'split': opt.val_split,
                   'dataset': opt.input_json}
    eval_kwargs.update(vars(opt))

    start_beam = 0
    total_beam = 20
    for beam in range(start_beam, total_beam):
        opt.beam_size = beam + 1
        eval_kwargs.update(vars(opt))
        print("beam_size: " + str(opt.beam_size))
        print("start eval ...")
        crit = None
        val_loss, predictions, lang_stats, str_stats = eval_utils.eval_split(model_cnn, model, crit, loader,
                                                                             eval_kwargs)
        print("end eval ...")
        msg = "str_stats = {}".format(str_stats)
        print(msg)
        save_result(str(opt.beam_size) + "," + str_stats, predictions, opt)


if __name__ == '__main__':
    main()