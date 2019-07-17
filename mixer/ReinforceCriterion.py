import torch
import torch.nn as nn
from torch.autograd import Variable
import RewardFactory

# This criterion implements the REINFORCE algorithm under the assumption that
# the reward does not depend on the model parameters.
# The constructor takes as input a function which is used to compute the reward
# given the ground truth input sequence, the generated sequence and the current
# time step.
# The input to the criterion is a table whose entries are the output of the
# RNN at a certain time step, namely:
# (chosen_word, predicted_cumulative_reward)_t
# It computes the total reward and bprop the derivative
# w.r.t. the above provided inputs.
#  reward_func: user provided function to compute the reward
#   given ground truth, current sequence and current time step.
# seq_length is the length of the sequence we use
# skips is the number of time steps we skip from the input and target (init)
# weight is the weight on the loss produced by this criterion
# weight_predictive_reward is the weight on the gradient of the cumulative
#   reward predictor (only)
class ReinforceCriterion(nn.Module):
    def __init__(self):
        super(ReinforceCriterion, self).__init__()

        self.sizeAverage = False
        self.skips = 0

        self.weight_predictive_reward = 0.01
        self.weight = 1
        self.num_samples = 0
        self.normalizing_coeff = 1

        self.reward_func = RewardFactory()

    def forward(self, input, target):
        # truncate to the same size
        # input (batch_size * (seq_length + 2) * (vocab_size + 1))
        # target (batch_size * (seq_length))
        batch_size, L, Mp1 = input.size(0), input.size(1), input.size(2)
        seq_length = target.size(1)

        cumreward = Variable(torch.FloatTensor(1).zero_(), requires_grad=True).cuda()

        for tt in xrange(seq_length):
            #
            reward = self.reward_func(input, target, tt)

            cumreward.add_(reward)

        # num_samples
        self.num_samples = self.reward_func.num_samples(input, target)

        # normalizing_coeff
        self.normalizing_coeff = self.weight / (self.sizeAverage and self.num_samples or 1)

        # here there is a '-' because we minimize
        self.output = -cumreward * self.normalizing_coeff

        # cumreward
        return self.output, self.num_samples