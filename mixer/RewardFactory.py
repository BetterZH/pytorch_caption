import torch
import torch.nn as nn
import math

# This class returns an object for computing the reward at a
# given time step.
# reward_type  type of reward, either ROUGE or BLEU
# start  index of time step at which we start computing the reward
# bptt   the maximum length of a sequence.
# dict_size  size of the dictionary
# eos_indx is the id of the end of sentence token. Symbols after
#  the first occurrence of eos (if any) are skipped.
# pad_indx is the id of the padding token
# mbsz mini-batch size

class RewardFactory(nn.Module):

    def __init__(self):
        super(RewardFactory, self).__init__()

    def forward(self, input, target, tt):
        pass

    def num_samples(self):
        pass

