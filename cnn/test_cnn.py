import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

linears = []
for i in range(1):
    linear = nn.Linear(10, 10)
    linear.weight.data.fill_(0)
    linear.bias.data.fill_(1)
    linears.append(linear)


# linear = nn.Linear(10, 1)
# linear.weight.data.fill_(0)
# linear.bias.data.fill_(1)
# linears.append(linear)

linears.append(nn.Softmax())


net = nn.Sequential(
    *linears
)

input = Variable(torch.FloatTensor(1, 10).zero_(), requires_grad=True)
output = net(input)

target = Variable(torch.LongTensor(1).fill_(2))

optimizer = optim.Adam(net.parameters())
crit = nn.CrossEntropyLoss()

loss = crit(output, target)
loss.backward()
optimizer.step()

for group in optimizer.param_groups:
    group_params = group['params']
    for ind in xrange(len(group_params)):
        param = group_params[ind]
        if param.grad is None:
            print('clip_error', param.size())
        else:
            print(param.grad.data)

