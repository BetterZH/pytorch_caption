import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

"""
acts: Tensor of (seqLength x batch x outputDim) containing output from network
labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
act_lens: Tensor of size (batch) containing size of each output sequence from the network
label_lens: Tensor of (batch) containing label length of each example
"""

ctc_loss = CTCLoss()
# expected shape of seqLength x batchSize x alphabet_size
probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()

print(probs.size())

labels = Variable(torch.IntTensor([0, 4]))
label_sizes = Variable(torch.IntTensor([2]))
probs_sizes = Variable(torch.IntTensor([2]))
probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs

cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
cost.backward()

print(cost)