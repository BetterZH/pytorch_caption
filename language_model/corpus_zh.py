from data_zh import *
from torch.autograd import Variable

data_path = 'data'
filename = 'sanguoyanyi.txt'

corpus = Corpus(data_path, filename)

print(len(corpus.dictionary))

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

train_data = batchify(corpus.train, 10)
print(train_data.size())

w2i = corpus.dictionary.word2idx
i2w = corpus.dictionary.idx2word

def get_batch(source, i, evaluation=False):
    seq_len = min(30, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

