import torch
import torch.nn as nn
from torch.autograd import Variable

# Module that takes a tensor storing log-probabilities (output of a LogSoftmax)
# and samples from the corresponding multinomial distribtion.
# Assumption: this receives input from a LogSoftMax and receives gradients from
# a ReinforceCriterion.
class ReinforceSampler(nn.Module):

    def __init__(self, distribution):
        super(ReinforceSampler, self).__init__()

        self.distribution = distribution
        self.prob = torch.FloatTensor()

    def forward(self, input):

        if self.distribution == 'multinomial':
            self.prob.resize_as(input)
            self.prob.copy_(input)
            self.prob.exp_()
            self.output.resize(input.size(0), 1)
            output = torch.multinomial(self.prob, 1)

        return output
