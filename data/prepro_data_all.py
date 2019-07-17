"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
from random import shuffle, seed
import string
import h5py
import numpy as np
from scipy.misc import imread, imresize
import time

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
    print '\n'.join(map(str, cw[:20]))

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

    def has_word(w, txt):
        if w in txt:
            return 1
        else:
            return 0

    N = len(imgs)
    for i, img in enumerate(imgs):
        start = time.time()
        img['final_captions'] = []
        # img['final_tokens'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

            # token = [has_word(w, txt) for w in vocab]
            # token.insert(0, 1)
            # img['final_tokens'].append(token)
        if i % 10000 == 0:
            print '%2d: %10d time: %.3fs' % (i, N, time.time() - start)


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
    for i, img in enumerate(imgs):
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

        if i % 10000 == 0:
            print '%2d: %10d' % (i, N)


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


def write_json(imgs, params, itow):
    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg['split'] = img['split']

        # if jimg['split'] == 'val' or jimg['split'] == 'test':
        #     jimg['split'] = 'train'

        if 'filename' in img and 'filepath' in img:
            jimg['file_path'] = os.path.join(img['filepath'], img['filename'])  # copy it over, might need
        elif 'filename' in img:
            jimg['file_path'] = img['filename']  # copy it over, might need

        # if 'cocoid' in img:
        #     jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)
        # elif 'imgid' in img:
        #     jimg['id'] = img['imgid']

        jimg['id'] = i

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

def write_anno(imgs, params):
    annos = {}
    annos['info'] = {}
    annos['licenses'] = []
    annos['type'] = "captions"
    annos['images'] = []
    annos['annotations'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        # if 'cocoid' in img:
        #     jimg['id'] = img['cocoid']
        # elif 'imgid' in img:
        #     jimg['id'] = img['imgid']

        jimg['id'] = i

        annos['images'].append(jimg)

        for sent in img['sentences']:
            janno = {}
            janno['image_id'] = jimg['id']
            janno['id'] = sent['sentid']
            janno['caption'] = sent['raw']
            annos['annotations'].append(janno)

    json.dump(annos, open(params['output_anno'], 'w'))
    print 'wrote ', params['output_anno']

def stat_count(imgs):

    train_count = 0
    val_count = 0
    test_count = 0

    for img in imgs:
        if img['split'] == 'train':
            train_count += 1
        elif img['split'] == 'val':
            val_count += 1
        elif img['split'] == 'test':
            test_count += 1

    print('train_count :', train_count)
    print('val_count :', val_count)
    print('test_count :', test_count)

def main(params):

    imgs = []
    first = True
    for input_json in params['input_json_paths']:
        print(input_json)
        sub_imgs = json.load(open(input_json, 'r'))
        sub_imgs = sub_imgs['images']

        if first:
            first = False
            for i, img in enumerate(sub_imgs):
                if img['split'] != 'test':
                    img['split'] = 'train'
        else:
            for i, img in enumerate(sub_imgs):
                img['split'] = 'train'

        for i, img in enumerate(sub_imgs):
            if 'flickr8k' in input_json:
                img['filepath'] = 'flickr8k'
            elif 'flickr30k' in input_json:
                img['filepath'] = 'flickr30k'

        imgs.extend(sub_imgs)

    # seed(123)  # make reproducible

    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi, vocab)

    # write to h5 file
    write_h5(imgs, params, L, label_start_ix, label_end_ix, label_length)

    # write to json file
    write_json(imgs, params, itow)

    # write to anno file
    write_anno(imgs, params)

    # stat count
    stat_count(imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    # http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    parser.add_argument('--root_path', type=str, default='/home/amds/caption/dataset/caption_datasets', help='input json file to process into hdf5')
    parser.add_argument('--input_jsons', type=list, default=['dataset_coco.json','dataset_flickr8k.json','dataset_flickr30k.json'], help='input json file to process into hdf5')

    parser.add_argument('--output_json', type=str, default='data_all.json', help='output json file')
    parser.add_argument('--output_h5', type=str, default='data_all.h5', help='output h5 file')
    parser.add_argument('--output_anno', type=str, default='anno_all.json', help='output anno file')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    params['input_json_paths'] = []
    for input_json in params['input_jsons']:
        params['input_json_paths'].append(os.path.join(params['root_path'], input_json))

    params['output_json'] = os.path.join(params['root_path'], params['output_json'])
    params['output_h5'] = os.path.join(params['root_path'], params['output_h5'])
    params['output_anno'] = os.path.join(params['root_path'], params['output_anno'])

    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)