import torch.nn as nn
import torch.nn.functional as F
import torch
import mixer.ReinforceSampler as ReinforceSampler
from torch.autograd import *
import torch.nn.init as init
import math
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class EmbeddingWithBias(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingWithBias, self).__init__()

        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):

        results = []
        for i in range(input.size(0)):
            index = input[i]
            result = self.weight[index.data] + self.bias
            results.append(result)

        return torch.cat(results, 0)



class WordEmbed(nn.Module):
    def __init__(self, word_gram_num):
        super(WordEmbed, self).__init__()

        self.convs = nn.ModuleList()
        for i in range(word_gram_num):
            n = i + 1
            padding = int(math.ceil((n - 1) / 2.0))
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
            conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=n, padding=padding)
            init.xavier_normal(conv.weight)
            self.convs.append(conv)


    def forward(self, x):

        # batch_size = x.size(0)
        feat_size = x.size(1)

        x = x.unsqueeze(1)

        xs = []
        for conv in self.convs:
            c = conv(x)
            c = c[:,:,:feat_size]
            xs.append(c)

        # batch_size * conv_num * feat_size
        x = torch.cat(xs, 1)

        x, _ = x.max(1)
        x = x.squeeze()

        x = F.tanh(x)
        x = F.dropout(x)

        return x

class PhraseEmbed(nn.Module):
    def __init__(self, phrase_gram_num, embed_size):
        super(PhraseEmbed, self).__init__()

        self.convs = nn.ModuleList()
        for i in range(phrase_gram_num):
            n = i + 1
            padding = int(math.ceil((n - 1) / 2.0))
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
            conv = nn.Conv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=n, padding=padding)
            init.xavier_normal(conv.weight)
            self.convs.append(conv)

    # input: batch_size * seq_length * embed_size
    # output: batch_size * embed_size
    def forward(self, x):

        # batch_size = x.size(0)
        seq_length = x.size(1)

        # batch_size * embed_size * seq_length
        x = x.transpose(1,2).contiguous()

        xs = []
        for conv in self.convs:
            c = conv(x)
            c = c[:,:,:seq_length]
            xs.append(c)

        # batch_size * conv_num * embed_size * seq_length
        x = torch.cat([_.unsqueeze(1) for _ in xs], 1)

        # batch_size * embed_size * seq_length
        x, _ = x.max(1)
        x = x.squeeze()

        x = F.tanh(x)
        x = F.dropout(x)

        # batch_size * embed_size
        x, _ = x.max(2)
        x = x.squeeze()

        return x

class ConvEmbed(nn.Module):
    def __init__(self, conv_gram_num):
        super(ConvEmbed, self).__init__()

        self.convs = nn.ModuleList()
        for i in range(conv_gram_num):
            n = i + 1
            padding = int(math.ceil((n - 1) / 2.0))
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
            conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=n, padding=padding)
            init.xavier_normal(conv.weight)
            self.convs.append(conv)


    # input: batch_size * seq_length * embed_size
    # output: batch_size * embed_size
    def forward(self, x):

        # batch_size = x.size(0)
        seq_length = x.size(1)
        embed_size = x.size(2)

        # batch_size * embed_size * seq_length
        # batch_size * 1 * embed_size * seq_length
        x = x.transpose(1,2).unsqueeze(1).contiguous()

        xs = []
        for conv in self.convs:
            c = conv(x)
            c = c[:, :, :embed_size, :seq_length]
            xs.append(c)


        # batch_size * conv_num * embed_size * seq_length
        x = torch.cat(xs, 1)

        # batch_size * embed_size * seq_length
        x, _ = x.max(1)
        # x = x.mean(1)
        x = x.squeeze()

        x = F.tanh(x)
        x = F.dropout(x)

        # batch_size * embed_size
        x, _ = x.max(2)
        # x = x.mean(2)
        x = x.squeeze()

        return x


if __name__ == '__main__':

    # input = Variable(torch.randn(8,32).float().zero_())
    #
    # # print(input)
    #
    # embed = WordEmbed(4)
    #
    # output = embed(input)
    #
    # print(output)
    #
    # print(output.size())

    batch_size = 16
    it = torch.LongTensor(batch_size).zero_()
    it = Variable(it).cuda()

    print(it.size())

    embed = EmbeddingWithBias(10, 512).cuda()

    result = embed(it)

    print(result.size())

    result.sum().backward()

    print(embed.weight.grad)



