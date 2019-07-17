import torch
import torch.nn as nn
from torch.autograd import Variable
from data_zh import *

import time
import math

class LMConfiguration(object):
    rnn_type = 'LSTM'
    vocab_size = 5000
    embedding_dim = 512
    hidden_dim = 512
    n_layers = 2
    dropout = 0.5
    tied_weights = True

    max_len = 50
    learning_rate = 1
    train_batch_size = 100


class RNNModel(nn.Module):
    def __init__(self, config):
        super(RNNModel, self).__init__()

        dropout = config.dropout
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim
        tied_weights = config.tied_weights

        self.hidden_dim = hidden_dim = config.hidden_dim
        self.rnn_type = rnn_type = config.rnn_type
        self.n_layers = n_layers = config.n_layers

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            raise ValueError("""'rnn_type' error, use ['LSTM', 'GRU']""")

        self.decoder = nn.Linear(hidden_dim, vocab_size)

        if tied_weights:
            if embedding_dim != hidden_dim:
                raise ValueError('When using the tied falg, embedding_dim must be equal to hidden_dim')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def forward(self, inputs, hidden):
        embedded = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(embedded, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            # LSTM h0, c0
            return (Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()),
                    Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()))
        else:
            # GRU h0
            return Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_())

def batchify(data, bsz, use_cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data



def get_batch(source, i, config, evaluation=False):
    seq_len = min(config.max_len, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model, train_batch_size, train_data, config, lr, criterion, epoch, use_cuda, corpus):
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(train_batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, config.max_len)):
        data, targets = get_batch(train_data, i, config)

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, config.vocab_size), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        print_per_batch = 200
        if batch % 200 == 0 and batch > 0:
            cur_loss = total_loss[0] / print_per_batch
            elapsed = time.time() - start_time
            msg = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} |'
            print(msg.format(epoch, batch, len(train_data) // config.max_len, lr,
                             elapsed * 1000 / print_per_batch, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        if batch % 1000 == 0 and batch > 0:
            word_list = generate(model, use_cuda, config, corpus)
            print(''.join(word_list))

            model.zero_grad()
            torch.save(model.state_dict(), 'checkpoint.pth')

def generate(model, use_cuda, config, corpus, word_len=100):
    inputs = Variable(torch.rand(1, 1).mul(config.vocab_size).long(), volatile=True)
    if use_cuda:
        inputs.data = inputs.data.cuda()
    hidden = model.init_hidden(1)
    word_list = []
    for i in range(word_len):
        output, hidden = model(inputs, hidden)
        word_weights = output.squeeze().data.div(1).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        inputs.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        word_list.append(word)
    return word_list

def main():

    use_cuda = torch.cuda.is_available()

    corpus = Corpus('/home/amds/caption/dataset/aic/aic_train.txt')

    print('dictionary len', len(corpus.dictionary))
    print('train len', len(corpus.train))

    config = LMConfiguration()
    train_batch_size = config.train_batch_size
    train_data = batchify(corpus.train, train_batch_size, use_cuda)

    print('train_data', train_data.size())

    config.vocab_size = len(corpus.dictionary)
    model = RNNModel(config)

    path_checkpoint = 'checkpoint.pth'
    if os.path.exists(path_checkpoint):
        model.load_state_dict(torch.load(path_checkpoint))

    if use_cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    lr = config.learning_rate
    for epoch in range(1, 100):
        train(model, train_batch_size, train_data, config, lr, criterion, epoch, use_cuda, corpus)
        lr /= 4

if __name__ == '__main__':
    main()





