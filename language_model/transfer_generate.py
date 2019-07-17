#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import jieba
import os
import json
import argparse
from random import shuffle, seed
import string
import h5py
import numpy as np
from scipy.misc import imread, imresize
import time
from tqdm import tqdm
import hashlib

def cut0(caption):
    tokens = []
    for i in range(len(caption)):
        tokens.append(caption[i])
    return tokens

def cut1(caption):
    seq_list = jieba.cut(caption)  # 默认是精确模式
    tokens = []
    for token in seq_list:
        tokens.append(token)
    return tokens

def build_sentences(imgs_train, imgs_val, cut_type):

    if cut_type == 0:
        cut = cut0
    elif cut_type == 1:
        cut = cut1

    new_imgs = []

    len_imgs_train = len(imgs_train)
    len_imgs_val = len(imgs_val)
    len_imgs = len_imgs_train + len_imgs_val
    # len_imgs = 1000
    file = open("tokens.txt", "w")
    j = 0
    len_split = len_imgs_train + 25000
    for i in tqdm(range(len_imgs)):
        if i < len_imgs_train:
            img = imgs_train[i]
            filepath = 'ai_challenger_caption_train_20170902/caption_train_images_20170902'
        else:
            img = imgs_val[i-len_imgs_train]
            filepath = 'ai_challenger_caption_validation_20170910/caption_validation_images_20170910'

        if i < len_split:
            split = 'train'
        else:
            split = 'val'


        new_img = {}
        # print(annotation)
        image_id = img['image_id']
        captions = img['caption']

        sentences = []
        for caption in captions:
            tokens = cut(caption)
            # print(" ".join(tokens)
            # raw = " ".join(tokens)
            file.write(" ".join(tokens) + "\n")
            sentence = {}
            sentence['tokens'] = tokens
            sentence['anno'] = ' '.join(cut1(caption))
            sentence['imgid'] = i
            sentence['sentid'] = j
            j += 1
            sentences.append(sentence)

        new_img['imgid'] = i
        new_img['sentences'] = sentences
        new_img['split'] = split
        new_img['filepath'] = filepath
        new_img['filename'] = image_id
        new_imgs.append(new_img)

    print('build sentences success')

    file.close()

    return new_imgs


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    # print '\n'.join(map(str, cw[:20]))
    for i in range(20):
        print(cw[i][1])

    # with open('words.txt', 'w') as file:
    #     for i in tqdm(range(len(cw))):
    #         file.write(str(cw[i][0]) + ' ' + cw[i][1] + "\n")

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    print 'number of words in vocab would be %d' % (len(vocab),)
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words)

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print 'max length sentence in raw data: ', max_len
    print 'sentence length distribution (count, number of words):'
    sum_len = sum(sent_lengths.values())
    for i in xrange(max_len + 1):
        print '%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print 'inserting the special UNK token'
        vocab.append('UNK')

    print('vocab size: ', len(vocab))

    def has_word(w, txt):
        if w in txt:
            return 1
        else:
            return 0

    N = len(imgs)
    for i in tqdm(range(N)):
        img = imgs[i]
        img['final_captions'] = []
        # img['final_tokens'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            if len(caption) > 0:
                img['final_captions'].append(caption)

    return vocab

def get_transfer_probability(imgs, vocab_size, wtoi, cut_type, params):

    output_trans_prob = params['output_trans_prob']

    if cut_type == 0:
        cut = cut0
    elif cut_type == 1:
        cut = cut1

    trans_matrix = np.zeros([vocab_size + 1, vocab_size + 1], dtype=float)

    N = len(imgs)
    for i in tqdm(range(N)):
        img = imgs[i]
        final_captions = img['final_captions']
        for caption in final_captions:
            tokens = caption + ['。']
            # print(' '.join(tokens))
            for j in range(len(tokens) - 1):
                t1 = wtoi[tokens[j]]
                t2 = wtoi[tokens[j+1]]
                trans_matrix[t1][t2] += 1

    trans_prob = np.zeros([vocab_size + 1, vocab_size + 1], dtype=float)
    for i in range(trans_matrix.shape[0]):
        print(i, trans_matrix[i].sum())
        if trans_matrix[i].sum() == 0:
            print(i, trans_matrix[i].sum())
            trans_prob[i] = trans_matrix[i] * 0
        else:
            trans_prob[i] = trans_matrix[i] / trans_matrix[i].sum()

    np.savez(output_trans_prob, x=trans_prob)


def main(params):

    cut_type = params['cut_type']

    if os.path.exists(params['input_json_new']):
        imgs = json.load(open(params['input_json_new'], 'r'))
    else:
        # build tokens
        imgs_train = json.load(open(params['input_json_train'], 'r'))
        imgs_val = json.load(open(params['input_json_val'], 'r'))

        imgs = build_sentences(imgs_train, imgs_val, cut_type)
        json.dump(imgs, open(params['input_json_new'], 'w'))

    print('load input json success')

    # create the vocab
    vocab = build_vocab(imgs, params)

    vocab_size = len(vocab)

    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    itow[0] = '。'
    wtoi['。'] = 0

    get_transfer_probability(imgs, vocab_size, wtoi, cut_type, params)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    # http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    parser.add_argument('--root_path', default='/home/public/dataset/aic/', help='input json file to process into hdf5')
    parser.add_argument('--input_json_train', default='ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', help='input json file to process into hdf5')
    parser.add_argument('--input_json_val', default='ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', help='input json file to process into hdf5')

    parser.add_argument('--input_json_new', default='dataset_aic5.json', help='input json file to process into hdf5')

    parser.add_argument('--output_trans_prob', default='trans_prob.npz', help='input json file to process into hdf5')


    parser.add_argument('--cut_type', default=0, type=int, help='cut type')

    # options
    parser.add_argument('--max_length', default=50, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    params['input_json_train'] = os.path.join(params['root_path'], params['input_json_train'])
    params['input_json_val'] = os.path.join(params['root_path'], params['input_json_val'])

    params['output_trans_prob'] = os.path.join(params['root_path'], params['output_trans_prob'])

    params['input_json_new'] = os.path.join(params['root_path'], params['input_json_new'])


    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)

