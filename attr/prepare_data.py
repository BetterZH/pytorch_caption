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
import nltk

def get_tag(word):
    words = []
    words.append(word)
    tag = nltk.pos_tag(words)[0]
    tag_key = tag[1]
    return tag_key


def is_tag_ok(tag):
    if tag in ['VB','VBG','VBD','VBN','VBP','VBZ','JJ','JJS','JJR','NN','NNS']:
        return True
    else:
        return False


def word_ok(word):
    return is_tag_ok(get_tag(word))

def build_vocab(imgs, params):

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str, cw[:1000]))


    vocab = []
    j = 0
    for i in xrange(len(cw)):
        word = cw[i][1]
        if word_ok(word):
            vocab.append(word)
            j += 1
            tag = get_tag(word)
            print(j, cw[i], tag)
            if j == 1000:
                break

    tags = {}
    for i in xrange(len(cw)):
        tag_key = get_tag(cw[i][1])
        tags[tag_key] = tags.get(tag_key, 0) + 1

    for tag_key in tags.keys():
        print(tag_key,tags[tag_key])

    # print some stats
    # total_words = sum(counts.itervalues())
    # print 'total words:', total_words
    # bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    # vocab = [w for w, n in counts.iteritems() if n > count_thr and word_ok(w)]
    # bad_count = sum(counts[w] for w in bad_words)
    # print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    # print 'number of words in vocab would be %d' % (len(vocab),)
    # print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words)

    # lets look at the distribution of lengths as well
    # sent_lengths = {}
    # for img in imgs:
    #     for sent in img['sentences']:
    #         txt = sent['tokens']
    #         nw = len(txt)
    #         sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    # max_len = max(sent_lengths.keys())
    # print 'max length sentence in raw data: ', max_len
    # print 'sentence length distribution (count, number of words):'
    # sum_len = sum(sent_lengths.values())
    # for i in xrange(max_len + 1):
    #     print '%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)

    # # lets now produce the final annotations
    # if bad_count > 0:
    #     # additional special UNK token we will use below to map infrequent words to
    #     print 'inserting the special UNK token'
    #     vocab.append('UNK')

    def has_word(w, txt):
        if w in txt:
            return 1
        else:
            return 0

    N = len(imgs)
    for i, img in enumerate(imgs):
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            # caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            caption = [has_word(w, txt) for w in vocab]
            img['final_captions'].append(caption)
        if i % 1000 == 0:
            print 'processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N)

    return vocab


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = w

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'

    print 'encoded captions to array of size ', `L.shape`
    return L, label_start_ix, label_end_ix


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    # imgs = imgs['images'][:1000]
    imgs = imgs['images']

    seed(123)  # make reproducible

    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix = encode_captions(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    dset = f.create_dataset("images", (N, 3, 256, 256), dtype='uint8')  # space for resized images

    for i, img in enumerate(imgs):
        # load the image
        I = imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
        try:
            Ir = imresize(I, (256, 256))
        except:
            print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
            print 'processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N)
    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img['filepath'],
                                                               img['filename'])  # copy it over, might need
        if 'cocoid' in img: jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    # http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    parser.add_argument('--input_json', default='/media/amds/data1/dataset/caption_datasets/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='/media/amds/data1/dataset/coco1/data_attr.json', help='output json file')
    parser.add_argument('--output_h5', default='/media/amds/data1/dataset/coco1/data_attr.h5', help='output h5 file')
    # options
    parser.add_argument('--max_length', default=1000, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='/media/amds/data3/dataset/mscoco',
                        help='root location in which images are stored, to be prepended to file_path in input json')


    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)