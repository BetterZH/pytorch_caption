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
        start = time.time()
        img['final_captions'] = []
        # img['final_tokens'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            if len(caption) > 0:
                img['final_captions'].append(caption)

            # if len(caption) == 0:
            #     print(img)

            # token = [has_word(w, txt) for w in vocab]
            # token.insert(0, 1)
            # img['final_tokens'].append(token)
        # if i % 10000 == 0:
        #     print '%2d: %10d time: %.3fs' % (i, N, time.time() - start)


    return vocab

def encode_captions(imgs, params, wtoi, vocab):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions
    vocab_len = len(vocab) + 1

    label_arrays = []
    label_tokens = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i in tqdm(range(N)):
        img = imgs[i]
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        # Lt = np.zeros((n, vocab_len), dtype='uint32')
        # for j, s in enumerate(img['final_tokens']):
        #     for k, w in enumerate(s):
        #         Lt[j, k] = w

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        # label_tokens.append(Lt)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

        # if i % 10000 == 0:
        #     print '%2d: %10d' % (i, N)


    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    # T = np.concatenate(label_tokens, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print 'encoded captions to array of size ', `L.shape`
    return L, label_start_ix, label_end_ix, label_length


def write_h5(imgs, params, L, label_start_ix, label_end_ix, label_length):
    # create output h5 file
    N = len(imgs)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("labels", dtype='uint32', data=L)
    # f.create_dataset("tokens", dtype='uint32', data=T)
    f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    f.close()
    print 'wrote ', params['output_h5']


def write_json(imgs, params, itow, wtoi):
    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['word_to_ix'] = wtoi  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg['split'] = img['split']

        jimg['image_id'] = img['filename'].split('.')[0]

        if 'filename' in img and 'filepath' in img:
            jimg['file_path'] = os.path.join(img['filepath'], img['filename'])  # copy it over, might need
        elif 'filename' in img:
            jimg['file_path'] = img['filename']  # copy it over, might need

        if 'cocoid' in img:
            jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)
        elif 'imgid' in img:
            jimg['id'] = img['imgid']

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

def get_hash(img_name):
    image_hash = int(int(hashlib.sha256(img_name).hexdigest(), 16) % sys.maxint)
    return image_hash

def write_anno(imgs, params):
    annos = {}
    annos['info'] = {}
    annos['licenses'] = []
    annos['type'] = "captions"
    annos['images'] = []
    annos['annotations'] = []
    len_imgs = len(imgs)

    for i in tqdm(range(len_imgs)):

        img = imgs[i]
        jimg = {}
        # if 'cocoid' in img:
        #     jimg['id'] = img['cocoid']
        # elif 'imgid' in img:
        #     jimg['id'] = img['imgid']

        file_name = img['filename']
        file_name = file_name.split('.')[0]

        image_id = get_hash(file_name)

        # print(image_id,file_name)

        jimg['id'] = image_id
        jimg['file_name'] = file_name
        annos['images'].append(jimg)

        for sent in img['sentences']:
            janno = {}
            janno['image_id'] = int(image_id)
            janno['id'] = int(sent['sentid'])
            janno['caption'] = sent['anno']
            annos['annotations'].append(janno)

    json.dump(annos, open(params['output_anno'], 'w'))
    print 'wrote ', params['output_anno']


def main(params):

    # imgs_train = json.load(open(params['input_json_train'], 'r'))
    # imgs_val = json.load(open(params['input_json_val'], 'r'))
    #
    # imgs = build_sentences(imgs_train, imgs_val)
    #
    # return

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
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi, vocab)

    # write to h5 file
    write_h5(imgs, params, L, label_start_ix, label_end_ix, label_length)

    # write to json file
    write_json(imgs, params, itow, wtoi)

    # write to anno file
    write_anno(imgs, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    # http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    parser.add_argument('--root_path', default='/home/amds/caption/dataset/aic', help='input json file to process into hdf5')
    parser.add_argument('--input_json_train', default='ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', help='input json file to process into hdf5')
    parser.add_argument('--input_json_val', default='ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', help='input json file to process into hdf5')

    parser.add_argument('--input_json_new', default='dataset_aic5.json', help='input json file to process into hdf5')

    parser.add_argument('--output_json', default='data_aic5.json', help='output json file')
    parser.add_argument('--output_h5', default='data_aic5.h5', help='output h5 file')
    parser.add_argument('--output_anno', default='anno_aic5.json', help='output anno file')

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

    params['input_json_new'] = os.path.join(params['root_path'], params['input_json_new'])
    params['output_json'] = os.path.join(params['root_path'], params['output_json'])
    params['output_h5'] = os.path.join(params['root_path'], params['output_h5'])
    params['output_anno'] = os.path.join(params['root_path'], params['output_anno'])

    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)

