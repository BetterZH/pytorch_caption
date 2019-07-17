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
    if tag in ['NN','NNS','NNP','NNPS']:
        return True
    else:
        return False

def word_ok(word):
    return is_tag_ok(get_tag(word))


def stat_vocab(imgs, params):
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

    tags = {}
    for i in xrange(len(cw)):
        tag_key = get_tag(cw[i][1])
        if tag_key not in tags.keys():
            tags[tag_key] = 0
        tags[tag_key] += 1

    for tag_key in tags.keys():
        print(tag_key,tags[tag_key])

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr and word_ok(w)]
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
    total_len = 0
    for i in xrange(max_len + 1):
        total_len += sent_lengths.get(i, 0)
        print '%2d: %10d   %f%% %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len, total_len * 100.0 / sum_len)

    return vocab


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']

    # create the vocab
    vocab = stat_vocab(imgs, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    # http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    parser.add_argument('--input_json', default='/home/scw4750/caption/dataset/mscoco/dataset_coco.json', help='input json file to process into hdf5')
    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='/home/scw4750/dataset',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')


    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)