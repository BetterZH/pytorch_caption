import torch
import gym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.autograd as autograd
import os
import cPickle
from torch.autograd import Variable

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 1  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = True

path_model = 'model.pth'
path_infos = 'infos.pkl'

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(D, H)
        self.linear2 = nn.Linear(H, 2)

        self.saved_probs = []
        self.rewards = []

        init.xavier_normal(self.linear1.weight)
        init.xavier_normal(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        p = F.softmax(x)
        return p



def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).cuda()
    probs = model(Variable(state))
    prob = probs.multinomial()
    model.saved_probs.append(prob)
    return prob.data

def finish_episode():
    R = 0
    rewards = []
    for r in model.rewards[::-1]:
        if r != 0:
            R = 0
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for prob, r in zip(model.saved_probs, rewards):
        prob.reinforce(r)
    optimizer.zero_grad()
    grad_variables = [None for _ in model.saved_probs]
    autograd.backward(model.saved_probs, grad_variables)
    optimizer.step()
    del model.rewards[:]
    del model.saved_probs[:]

model = Policy()
model.cuda()

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
running_reward = None
reward_sum = 0
episode_number = 0
start = time.time()


if os.path.exists(path_model):
    model.load_state_dict(torch.load(path_model))

if os.path.exists(path_infos):
    with open(path_infos) as f:
        infos = cPickle.load(f)
        episode_number = infos['episode_number']



optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

while True:

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    prob = select_action(x)

    action = (prob[0,0] == 0 and 2 or 3)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)

    if render:
        env.render()

    model.rewards.append(reward)

    reward_sum += reward

    if done:  # an episode finished

        episode_number += 1

        if episode_number % batch_size == 0:
            finish_episode()

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'episode %d reward total was %f. running mean: %f time: %.2f' % (episode_number, reward_sum, running_reward, time.time()-start)
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
        start = time.time()

        torch.save(model.state_dict(), path_model)

        infos = {}
        infos['episode_number'] = episode_number
        with open(path_infos, 'wb') as f:
            cPickle.dump(infos, f)


    # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #     print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

