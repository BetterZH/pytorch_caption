from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import misc.utils as utils
import numpy as np
import torch.autograd as autograd
import vis.visual_rnn as visual_rnn

# Fast
class CTCCriterion(nn.Module):
    def __init__(self):
        super(CTCCriterion, self).__init__()
        from warpctc_pytorch import CTCLoss
        self.ctc_loss = CTCLoss()

    # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
    # labels : batch_size * (seq_length + 1)  (the last is zero)
    def forward(self, input, labels, mask):

        # visual_rnn.show_all_probs(input[0].data.cpu().numpy())

        batch_size = input.size(0)
        seq_len = input.size(1)
        labels_len = labels.size(1)

        # (seq_length + 1) * batch_size * (vocab_size + 1)
        input = input.transpose(0, 1).contiguous()

        # batch_size * (seq_length + 1)
        labels = labels.int().view(-1).cpu()

        seq_lens = []
        for i in range(batch_size):
            mask_one = mask[i]
            mask_len = mask_one[mask_one>0].size(0)
            seq_lens.append(mask_len)

        probs_sizes = Variable(torch.IntTensor(seq_lens))
        label_sizes = Variable(torch.IntTensor(seq_lens))

        loss = self.ctc_loss(input, labels, probs_sizes, label_sizes)

        return loss

# Fast
# class LanguageModelWeightNewCriterion(nn.Module):
#     def __init__(self):
#         super(LanguageModelWeightNewCriterion, self).__init__()
#
#     # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
#     # target : batch_size * (seq_length + 1)  (the last is zero)
#     def forward(self, input, target, mask):
#         # target = target[:,:input.size(1)].contiguous()
#         # mask = mask[:, :input.size(1)].contiguous()
#
#         # visual_rnn.show_all_probs(input[0].data.cpu().numpy())
#
#         # (batch_size * (seq_length + 1)) * (vocab_size + 1)
#         input = input.view(-1, input.size(2))
#
#         # (batch_size * (seq_length + 1)) * 1
#         target = target.view(-1, 1)
#
#         # (batch_size * (seq_length + 1))
#         output = - input.gather(1, target) * mask
#
#         # average loss
#         loss = output.sum() / mask[mask > 0].size(0)
#
#         return loss

# Fast
class LanguageModelWeightMulOutWithWeightCriterion(nn.Module):
    def __init__(self, alpha):
        super(LanguageModelWeightMulOutWithWeightCriterion, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha

    # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
    # target : batch_size * (seq_length + 1)  (the last is zero)
    def forward(self, inputs, target, mask, prob_w, token):
        losses = []
        len_input = len(inputs)
        for i in range(len_input):
            input = inputs[i]

            target1 = target[:, :input.size(1)].contiguous()
            mask1 = mask[:, :input.size(1)].contiguous()

            # (batch_size * (seq_length + 1)) * (vocab_size + 1)
            input = input.view(-1, input.size(2))

            # (batch_size * (seq_length + 1)) * 1
            target1 = target1.view(-1, 1)

            # (batch_size * (seq_length + 1))
            output = - input.gather(1, target1) * mask1

            # average loss
            loss = output.sum() / mask1[mask1 > 0].size(0)

            # 1
            losses.append(loss)

        loss = torch.cat([_ for _ in losses]).sum()

        # BCE loss
        # neg_abs = - prob_w.abs()
        # loss_bce = prob_w.clamp(min=0) - prob_w * target + (1 + neg_abs.exp()).log()
        loss_bce = self.bce_loss(prob_w, token)

        print('loss:{:.3f} loss_bce:{:.3f}'.format(loss.data[0], loss_bce.data[0]))

        return loss * self.alpha + loss_bce * (1 - self.alpha)


# Fast
class LanguageModelWeightMulOutCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelWeightMulOutCriterion, self).__init__()

    # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
    # target : batch_size * (seq_length + 1)  (the last is zero)
    # mask   : mask * (seq_length + 1)
    def forward(self, inputs, target, mask):

        losses = []
        for i in range(len(inputs)):

            input = inputs[i]

            target1 = target[:, :input.size(1)].contiguous()
            mask1 = mask[:, :input.size(1)].contiguous()

            # (batch_size * (seq_length + 1)) * (vocab_size + 1)
            input = input.view(-1, input.size(2))

            # (batch_size * (seq_length + 1)) * 1
            target1 = target1.view(-1, 1)

            # (batch_size * (seq_length + 1))
            output = - input.gather(1, target1).view_as(mask1) * mask1

            # average loss
            loss = output.sum() / mask1[mask1 > 0].size(0)

            # 1
            losses.append(loss)

        loss = torch.cat([_ for _ in losses]).mean()

        return loss

# Fast With Weight
class LanguageModelWithProbWeightCriterion(nn.Module):
    def __init__(self, alpha):
        super(LanguageModelWithProbWeightCriterion, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha

    # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
    # target : batch_size * (seq_length + 1)  (the last is zero)
    def forward(self, input, target, mask, prob_w, token):

        target = target[:, :input.size(1)].contiguous()
        mask = mask[:, :input.size(1)].contiguous()

        # (batch_size * (seq_length + 1)) * (vocab_size + 1)
        input = input.view(-1, input.size(2))

        # (batch_size * (seq_length + 1)) * 1
        target = target.view(-1, 1)

        # (batch_size * (seq_length + 1))
        output = - input.gather(1, target).view_as(mask) * mask

        # average loss
        loss = output.sum() / mask[mask > 0].size(0)

        # BCE loss
        # neg_abs = - prob_w.abs()
        # loss_bce = prob_w.clamp(min=0) - prob_w * target + (1 + neg_abs.exp()).log()

        if prob_w.size() == token.size():
            loss_bce = self.bce_loss(prob_w, token)
        else:
            loss_bce = self.bce_loss(prob_w, token.unsqueeze(1).expand_as(prob_w))
            # loss_bce = self.bce_loss(prob_w.mean(1), token)

        print('loss:{:.3f} loss_bce:{:.3f}'.format(loss.data[0], loss_bce.data[0]))

        return loss * self.alpha + loss_bce * (1 - self.alpha)


# Fast With Weight and Mul Out
class LanguageModelWithProbWeightMulOutCriterion(nn.Module):
    def __init__(self, alpha):
        super(LanguageModelWithProbWeightMulOutCriterion, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha

    # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
    # target : batch_size * (seq_length + 1)  (the last is zero)
    def forward(self, inputs, target, mask, prob_ws, token):

        losses = []

        for i in range(len(inputs)):

            input = inputs[i]
            prob_w = prob_ws[i]

            input_size = input.size(1)

            target1 = target[:, :input.size(1)].contiguous()
            mask1 = mask[:, :input.size(1)].contiguous()

            # (batch_size * (seq_length + 1)) * (vocab_size + 1)
            input = input.view(-1, input.size(2))

            # (batch_size * (seq_length + 1)) * 1
            target1 = target1.view(-1, 1)

            # (batch_size * (seq_length + 1))
            output = - input.gather(1, target1) * mask1

            # average loss
            loss = output.sum() / mask1[mask1 > 0].size(0)

            # bce loss
            loss_bce = self.bce_loss(prob_w, token)

            print('{:d} input size: {:d} loss:{:.3f} loss_bce:{:.3f}'.format(i, input_size, loss.data[0], loss_bce.data[0]))

            final_loss = loss * self.alpha + loss_bce * (1 - self.alpha)

            losses.append(final_loss)

        all_loss = torch.cat([_ for _ in losses]).mean()

        return all_loss

# Fast
class LanguageModelWeightCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelWeightCriterion, self).__init__()

    # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
    # target : batch_size * (seq_length + 1)  (the last is zero)
    def forward(self, input, target, mask):

        target = target[:,:input.size(1)].contiguous()
        mask = mask[:, :input.size(1)].contiguous()

        # (batch_size * (seq_length + 1)) * (vocab_size + 1)
        input = input.view(-1, input.size(2))

        # (batch_size * (seq_length + 1)) * 1
        target = target.view(-1, 1)

        # (batch_size * (seq_length + 1))
        output = - input.gather(1, target).view_as(mask) * mask

        # average loss
        loss = output.sum() / mask[mask>0].size(0)

        return loss

# Fast
# class LanguageModelCriterion(nn.Module):
#     def __init__(self):
#         super(LanguageModelCriterion, self).__init__()
#
#     # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
#     # target : batch_size * (seq_length + 1)  (the last is zero)
#     def forward(self, input, target, mask):
#
#         target = target[:,:input.size(1)].contiguous()
#         mask = mask[:, :input.size(1)].contiguous()
#
#         # (batch_size * (seq_length + 1)) * (vocab_size + 1)
#         input = input.view(-1, input.size(2))
#
#         # (batch_size * (seq_length + 1)) * 1
#         target = target.view(-1, 1)
#
#         # (batch_size * (seq_length + 1)) * 1
#         mask = mask.view(-1, 1)
#
#         # (batch_size * (seq_length + 1))
#         output = - input.gather(1, target) * mask
#
#         # average loss
#         loss = output.sum() / mask.sum()
#
#         return loss


# NLL
class LanguageModelOldCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelOldCriterion, self).__init__()

    def forward(self, input, target):
        # truncate to the same size
        # input (batch_size * (seq_length + 2) * (vocab_size + 1))
        # target (batch_size * (seq_length))
        batch_size, L, Mp1 = input.size(0), input.size(1), input.size(2)
        seq_length = target.size(1)

        loss = Variable(torch.FloatTensor(1).zero_(),requires_grad=True).cuda()
        n = 0

        for b in range(batch_size):
            first_time = True
            for t in range(1, L):

                if t - 1 >= seq_length:
                    target_index = 0
                else:
                    target_index = target.data[b, t-1]

                if target_index == 0 and first_time:
                    first_time = False
                elif target_index == 0 and not first_time:
                    break

                logsoft = input[b, t, target_index]
                loss.sub_(logsoft)
                n += 1

        loss.div_(n)

        return loss

# RewardCriterion
# self-critical
class RewardCriterion1(nn.Module):
    def __init__(self, opt):
        super(RewardCriterion1, self).__init__()
        self.opt = opt
        self.reward_total = 0
        self.reward_num = 0

    # sample_seq    batch_size * seq_length
    # seqLogprobs   batch_size * seq_length
    # seq1          batch_size * seq_length
    # seqLogprobs1  batch_size * seq_length
    # target        batch_size * (seq_length + 1)  (the last is zero)
    # sample_seqLogprobs batch_size * seq_length * (vocab_size + 1)
    def forward(self, sample_seq, sample_seqLogprobs, gts):
        # greedy_seq : batch_size * seq_length
        # sample_seq : batch_size * seq_length
        # reward_diff : batch_size * seq_length
        reward_sample, sample_mean = utils.get_reward_cirder(sample_seq, gts, self.opt)

        self.reward_total += sample_mean
        self.reward_num += 1

        print("Ave Reward {:.3f}".format(self.reward_total / self.reward_num))

        mask = (sample_seq > 0).float()

        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

        reward_sample = Variable(torch.from_numpy(reward_sample).float().cuda(), requires_grad=False)

        # reward_diff = reward_sample - (self.reward_total / self.reward_num)

        reward_diff = reward_sample - 1.5

        # (batch_size * (seq_length + 1))
        output = - sample_seqLogprobs * reward_diff * Variable(mask)

        # seqLogprobs.reinforce(reward_diff)
        # output = - seqLogprobs * mask

        # average loss
        loss = output.sum() / mask.sum()

        return loss, reward_diff.mean().data[0]

# RewardCriterion
# self-critical
class RewardMulOutCriterionAIC(nn.Module):
    def __init__(self, opt, vocab):
        super(RewardMulOutCriterionAIC, self).__init__()
        self.opt = opt

        self.reward_sample_total = 0
        self.reward_greedy_total = 0
        self.reward_num = 0

        self.alpha_type = opt.rl_alpha_type
        self.alpha = opt.rl_alpha_start
        self.recent_alpha = opt.rl_alpha_recent_start
        self.recent_num = opt.rl_alpha_recent_num

        self.recent_alpha_list = np.linspace(0, 0, self.recent_num)
        self.recent_gamma_list = np.linspace(0, 0, self.recent_num)
        self.recent_index = 0

        self.beta = opt.rl_beta
        self.gamma = opt.rl_gamma
        self.use_gamma = opt.rl_use_gamma

        self.vocab = vocab

    # sample_seq    batch_size * seq_length
    # seqLogprobs   batch_size * seq_length
    # seq1          batch_size * seq_length
    # seqLogprobs1  batch_size * seq_length
    # target        batch_size * (seq_length + 1)  (the last is zero)
    # sample_seqLogprobs batch_size * seq_length * (vocab_size + 1)
    def forward(self, list_sample_seq, list_sample_seqLogprobs, greedy_seq, gts):

        all_loss = []
        all_reward_diff_mean = []
        all_sample_mean = []
        all_greedy_mean = []

        for i in range(len(list_sample_seq)):

            sample_seq = list_sample_seq[i]
            sample_seqLogprobs = list_sample_seqLogprobs[i]

            batch_size = sample_seq.size(0)
            seq_length = sample_seq.size(1)

            for i in range(batch_size):
                k = 0
                for j in range(seq_length):
                    if sample_seq[i, j] == 0:
                        k = 1
                    if k == 1:
                        sample_seq[i, j] = 0

            print("alpha {:.3f}, recent_alpha {:.3f}".format(self.alpha, self.recent_alpha))

            # greedy_seq : batch_size * seq_length
            # sample_seq : batch_size * seq_length
            # reward_diff : batch_size * seq_length

            if self.alpha_type == 0:
                temp_alpha = 1.0
            elif self.alpha_type == 1:
                temp_alpha = self.recent_alpha * self.beta
            elif self.alpha_type == 2:
                temp_alpha = self.alpha * self.beta

            reward_diff, sample_mean, greedy_mean = utils.get_self_critical_reward_aic(greedy_seq, sample_seq, gts,
                                                                                       temp_alpha, self.vocab, self.opt)

            self.reward_sample_total += sample_mean
            self.reward_greedy_total += greedy_mean
            self.reward_num += 1

            reward_sample_avg = self.reward_sample_total / self.reward_num
            reward_greedy_avg = self.reward_greedy_total / self.reward_num

            self.alpha = self.reward_sample_total / self.reward_greedy_total

            # recent num
            self.recent_alpha_list[self.recent_index % self.recent_num] = sample_mean / greedy_mean
            if sample_mean - greedy_mean * temp_alpha == 0:
                temp_gamma = 1
            else:
                temp_gamma = 1 / np.abs(sample_mean - temp_alpha * greedy_mean)

            self.recent_gamma_list[self.recent_index % self.recent_num] = temp_gamma
            self.recent_index += 1

            if self.recent_index <= self.recent_num:
                self.recent_alpha = np.mean(self.recent_alpha_list[:self.recent_index])
                self.recent_gamma = np.mean(self.recent_gamma_list[:self.recent_index])
            else:
                self.recent_alpha = np.mean(self.recent_alpha_list)
                self.recent_gamma = np.mean(self.recent_gamma_list)

            print("avg sample reward {:.3f}, avg greedy reward {:.3f} recent_gamma {:.3f}".format(reward_sample_avg,
                                                                                                  reward_greedy_avg,
                                                                                                  self.recent_gamma))

            # batch_size * seq_length
            mask = (sample_seq > 0).float()

            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

            reward_diff = Variable(torch.from_numpy(reward_diff).float().cuda(), requires_grad=False)

            if self.use_gamma:
                temp_gamma = self.recent_gamma * self.gamma
            else:
                temp_gamma = 1

            # (batch_size * (seq_length + 1))
            output = - sample_seqLogprobs * reward_diff * Variable(mask) * temp_gamma

            # seqLogprobs.reinforce(reward_diff)
            # output = - seqLogprobs * mask

            # average loss
            loss = output.sum() / mask.sum()

            # while loss.data[0] > 1 or loss.data[0] < -1:
            #     loss = loss * 0.1

            all_loss.append(loss)
            all_reward_diff_mean.append(reward_diff.mean().data[0])
            all_sample_mean.append(sample_mean)
            all_greedy_mean.append(greedy_mean)

        final_loss = torch.cat([_ for _ in all_loss]).mean()
        final_reward_diff_mean = np.array(all_reward_diff_mean).mean()
        final_sample_mean = np.array(all_sample_mean).mean()
        final_greedy_mean = np.array(all_greedy_mean).mean()

        return final_loss, final_reward_diff_mean, final_sample_mean, final_greedy_mean


# RewardCriterion
# self-critical
class RewardCriterionAIC(nn.Module):
    def __init__(self, opt, vocab):
        super(RewardCriterionAIC, self).__init__()
        self.opt = opt

        self.reward_sample_total = 0
        self.reward_greedy_total = 0
        self.reward_num = 0

        self.alpha_type = opt.rl_alpha_type
        self.alpha = opt.rl_alpha_start
        self.recent_alpha = opt.rl_alpha_recent_start
        self.recent_num = opt.rl_alpha_recent_num

        self.recent_alpha_list = np.linspace(0, 0, self.recent_num)
        self.recent_gamma_list = np.linspace(0, 0, self.recent_num)
        self.recent_index = 0

        self.gamma = opt.rl_gamma
        self.use_gamma = opt.rl_use_gamma

        self.vocab = vocab

        self.beta_incre_start = opt.rl_beta_incre_start
        self.beta_incre_iters_every = opt.rl_beta_incre_iters_every
        self.beta_incre_every_add = opt.rl_beta_incre_every_add
        self.beta_incre_every_rate = opt.rl_beta_incre_every_add
        self.beta_incre_max = opt.rl_beta_incre_max
        self.beta = opt.rl_beta
        self.beta_ini = opt.rl_beta
        self.use_beta_incre = opt.rl_beta_incre_start >= 0
        self.is_beta_incre_linear = opt.is_beta_incre_linear
        if self.use_beta_incre:
            print("## Use ingincresing beta!")
        self.done_iters = 0

        self.hard_type = opt.rl_hard_type
        self.hard_alpha = opt.rl_hard_alpha
        self.hard_reward = opt.rl_hard_reward

        self.mask_type = opt.rl_mask_type


    # sample_seq    batch_size * (seq_length+1)
    # sample_seqLogprobs   batch_size * (seq_length+1)
    # greedy_seq    batch_size * (seq_length+1)
    # gts        batch_size * (seq_length + 1)  (the last is zero)
    def forward(self, sample_seq, sample_seqLogprobs, greedy_seq, gts, mask):

        self.done_iters += 1
        if self.use_beta_incre and  self.done_iters > self.beta_incre_start and self.beta < self.beta_incre_max:
            if self.is_beta_incre_linear or (self.done_iters - self.beta_incre_start) % self.beta_incre_iters_every == 0:
                self.beta = min(self.beta_ini + self.beta_incre_every_add * (self.done_iters - self.beta_incre_start)/self.beta_incre_iters_every, self.beta_incre_max)
                print("####### Update beta {:.3f}".format(self.beta))

        batch_size = sample_seq.size(0)
        seq_length = sample_seq.size(1)

        for i in range(batch_size):
            k = 0
            for j in range(seq_length):
                if sample_seq[i, j] == 0:
                    k = 1
                if k == 1:
                    sample_seq[i, j] = 0

        print("alpha {:.3f}, recent_alpha {:.3f}".format(self.alpha, self.recent_alpha))

        # greedy_seq : batch_size * seq_length
        # sample_seq : batch_size * seq_length
        # reward_diff : batch_size * seq_length

        if self.alpha_type == 0:
            temp_alpha = 1.0
        elif self.alpha_type == 1:
            temp_alpha = self.recent_alpha * self.beta
        elif self.alpha_type == 2:
            temp_alpha = self.alpha * self.beta

        reward_diff, sample_mean, greedy_mean = utils.get_self_critical_reward_aic(greedy_seq, sample_seq, gts,
                                                                               temp_alpha, self.vocab, self.opt)

        self.reward_sample_total += sample_mean
        self.reward_greedy_total += greedy_mean
        self.reward_num += 1

        reward_sample_avg = self.reward_sample_total / self.reward_num
        reward_greedy_avg = self.reward_greedy_total / self.reward_num

        self.alpha = self.reward_sample_total / self.reward_greedy_total

        # recent num
        current_alpha = sample_mean / greedy_mean
        self.recent_alpha_list[self.recent_index % self.recent_num] = current_alpha
        if sample_mean - greedy_mean * temp_alpha == 0:
            temp_gamma = 1
        else:
            temp_gamma = 1 / np.abs(sample_mean - temp_alpha * greedy_mean)

        self.recent_gamma_list[self.recent_index % self.recent_num] = temp_gamma
        self.recent_index += 1

        if self.recent_index <= self.recent_num:
            self.recent_alpha = np.mean(self.recent_alpha_list[:self.recent_index])
            self.recent_gamma = np.mean(self.recent_gamma_list[:self.recent_index])
        else:
            self.recent_alpha = np.mean(self.recent_alpha_list)
            self.recent_gamma = np.mean(self.recent_gamma_list)

        print("avg sample reward {:.3f}, avg greedy reward {:.3f} recent_gamma {:.3f}".format(reward_sample_avg,
                                                                                              reward_greedy_avg,
                                                                                              self.recent_gamma))

        if self.mask_type == 0:
            # batch_size * seq_length
            mask = (sample_seq > 0).float()
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            mask = Variable(mask)
        elif self.mask_type == 1:
            # batch_size * seq_length
            mask = (greedy_seq > 0).float()
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            mask = Variable(mask)


        reward_diff = Variable(torch.from_numpy(reward_diff).float().cuda(), requires_grad=False)

        if self.use_gamma:
            temp_gamma = self.recent_gamma * self.gamma
        else:
            temp_gamma = 1

        # (batch_size * (seq_length + 1))
        output = - sample_seqLogprobs * reward_diff * mask * temp_gamma

        # seqLogprobs.reinforce(reward_diff)
        # output = - seqLogprobs * mask

        # average loss
        loss = output.sum() / mask.sum()

        # while loss.data[0] > 1 or loss.data[0] < -1:
        #     loss = loss * 0.1

        if self.hard_type == 1:
            print("current_alpha {:.3f} recent_alpha {:.3f}".format(current_alpha, self.recent_alpha))
            if current_alpha >= self.recent_alpha * self.hard_alpha and self.reward_num > 10:
                loss = loss * 0
        elif self.hard_type == 2:
            if sample_mean >= greedy_mean * self.hard_alpha:
                loss = loss * 0
        elif self.hard_type == 3:
            if reward_diff >= self.hard_reward:
                loss = loss * 0

        return loss, reward_diff.mean().data[0], sample_mean, greedy_mean

# RewardCriterion
# self-critical
class ActorCriticCriterionAIC(nn.Module):
    def __init__(self, opt, vocab):
        super(ActorCriticCriterionAIC, self).__init__()
        self.opt = opt
        self.vocab = vocab
        self.gamma = opt.rl_gamma

    # sample_seq          batch_size * (seq_length + 1)
    # sample_seqLogprobs  batch_size * (seq_length + 1)
    # sample_value        batch_size * (seq_length + 1)
    def forward(self, sample_seq, sample_seqLogprobs, sample_value, gts_data):

        batch_size = sample_seq.size(0)
        seq_length = sample_seq.size(1)

        for i in range(batch_size):
            k = 0
            for j in range(seq_length):
                if sample_seq[i, j] == 0:
                    k = 1
                if k == 1:
                    sample_seq[i, j] = 0

        # sample_reward : batch_size * (seq_length + 1)
        # sample_mean : 1
        sample_reward, sample_mean = utils.get_sample_reward_aic(sample_seq, gts_data, self.gamma, self.vocab, self.opt)
        print("avg sample reward {:.3f}".format(sample_mean))

        # batch_size * (seq_length + 1)
        mask = (sample_seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

        # sample_reward : batch_size * (seq_length + 1)
        # sample_value : batch_size * (seq_length + 1)
        sample_reward = Variable(torch.from_numpy(sample_reward).float().cuda())

        reward_diff = sample_reward - sample_value

        # (batch_size * (seq_length + 1))
        output = - sample_seqLogprobs * reward_diff * Variable(mask)

        # seqLogprobs.reinforce(reward_diff)
        # output = - seqLogprobs * mask

        # average loss
        loss = output.sum() / mask.sum()

        # while loss.data[0] > 1 or loss.data[0] < -1:
        #     loss = loss * 0.1

        return loss, reward_diff.mean().data[0], sample_mean

# RewardCriterion
# self-critical
class ActorCriticMSECriterionAIC(nn.Module):
    def __init__(self, opt, vocab):
        super(ActorCriticMSECriterionAIC, self).__init__()
        self.opt = opt
        self.vocab = vocab
        self.gamma = opt.rl_gamma

    # sample_seq          batch_size * (seq_length + 1)
    # sample_value        batch_size * (seq_length + 1) * 1
    def forward(self, sample_seq, sample_value, gts_data):

        batch_size = sample_seq.size(0)
        seq_length = sample_seq.size(1)

        for i in range(batch_size):
            k = 0
            for j in range(seq_length):
                if sample_seq[i, j] == 0:
                    k = 1
                if k == 1:
                    sample_seq[i, j] = 0

        # sample_reward : batch_size * (seq_length + 1)
        # sample_mean : 1
        sample_reward, sample_mean = utils.get_sample_reward_aic(sample_seq, gts_data, self.gamma, self.vocab,
                                                                 self.opt)

        print("avg sample reward {:.3f}".format(sample_mean))

        # batch_size * (seq_length + 1)
        mask = (sample_seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

        # sample_reward : batch_size * (seq_length + 1)
        # sample_value : batch_size * (seq_length + 1)

        sample_reward = Variable(torch.from_numpy(sample_reward).float().cuda())

        reward_diff = sample_reward - sample_value

        print("sample_reward {:.3f} sample_value {:.3f}".format(sample_reward.mean().data[0], sample_value.mean().data[0]))

        # (batch_size * (seq_length + 1))
        output = reward_diff * reward_diff * Variable(mask)

        # seqLogprobs.reinforce(reward_diff)
        # output = - seqLogprobs * mask

        # average loss
        loss = output.sum() / mask.sum()

        # while loss.data[0] > 1 or loss.data[0] < -1:
        #     loss = loss * 0.1

        return loss, reward_diff.mean().data[0], sample_mean


# RewardCriterion
# self-critical
class RewardCriterion(nn.Module):
    def __init__(self, opt):
        super(RewardCriterion, self).__init__()
        self.opt = opt

        self.reward_sample_total = 0
        self.reward_greedy_total = 0
        self.reward_num = 0

        self.alpha_type = opt.rl_alpha_type
        self.alpha = opt.rl_alpha_start
        self.recent_alpha = opt.rl_alpha_recent_start
        self.recent_num = opt.rl_alpha_recent_num

        self.recent_alpha_list = np.linspace(0, 0, self.recent_num)
        self.recent_gamma_list = np.linspace(0, 0, self.recent_num)
        self.recent_index = 0

        self.gamma = opt.rl_gamma
        self.use_gamma = opt.rl_use_gamma

        self.beta_incre_start = opt.rl_beta_incre_start
        self.beta_incre_iters_every = opt.rl_beta_incre_iters_every
        self.beta_incre_every_add = opt.rl_beta_incre_every_add
        self.beta_incre_every_rate = opt.rl_beta_incre_every_add
        self.beta_incre_max = opt.rl_beta_incre_max
        self.beta = opt.rl_beta
        self.beta_ini = opt.rl_beta
        self.use_beta_incre = opt.rl_beta_incre_start >= 0
        self.is_beta_incre_linear = opt.is_beta_incre_linear
        if self.use_beta_incre:
            print("## Use ingincresing beta!")
        self.done_iters = 0

        self.mask_type = opt.rl_mask_type


    # sample_seq    batch_size * seq_length
    # seqLogprobs   batch_size * seq_length
    # seq1          batch_size * seq_length
    # seqLogprobs1  batch_size * seq_length
    # target        batch_size * (seq_length + 1)  (the last is zero)
    # sample_seqLogprobs batch_size * seq_length * (vocab_size + 1)
    def forward(self, sample_seq, sample_seqLogprobs, greedy_seq, gts, mask):

        self.done_iters += 1
        if self.use_beta_incre and  self.done_iters > self.beta_incre_start and self.beta < self.beta_incre_max:
            if self.is_beta_incre_linear or (self.done_iters - self.beta_incre_start) % self.beta_incre_iters_every == 0:
                self.beta = min(self.beta_ini + self.beta_incre_every_add * (self.done_iters - self.beta_incre_start)/self.beta_incre_iters_every, self.beta_incre_max)
                print("####### Update beta {:.3f}".format(self.beta))

        batch_size = sample_seq.size(0)
        seq_length = sample_seq.size(1)

        for i in range(batch_size):
            k = 0
            for j in range(seq_length):
                if sample_seq[i,j] == 0:
                    k = 1
                if k == 1:
                    sample_seq[i,j] = 0

        print("alpha {:.3f}, recent_alpha {:.3f}".format(self.alpha, self.recent_alpha))

        # greedy_seq : batch_size * seq_length
        # sample_seq : batch_size * seq_length
        # reward_diff : batch_size * seq_length

        if self.alpha_type == 0:
            temp_alpha = 1.0
        elif self.alpha_type == 1:
            temp_alpha = self.recent_alpha * self.beta
        elif self.alpha_type == 2:
            temp_alpha = self.alpha * self.beta

        reward_diff, sample_mean, greedy_mean = utils.get_self_critical_reward(greedy_seq, sample_seq, gts, temp_alpha, self.opt)

        self.reward_sample_total += sample_mean
        self.reward_greedy_total += greedy_mean
        self.reward_num += 1

        reward_sample_avg = self.reward_sample_total / self.reward_num
        reward_greedy_avg = self.reward_greedy_total / self.reward_num

        self.alpha = self.reward_sample_total / self.reward_greedy_total

        # recent num
        self.recent_alpha_list[self.recent_index % self.recent_num] = sample_mean / greedy_mean
        if sample_mean - greedy_mean * temp_alpha == 0:
            temp_gamma = 1
        else:
            temp_gamma = 1 / np.abs(sample_mean - temp_alpha * greedy_mean)

        self.recent_gamma_list[self.recent_index % self.recent_num] = temp_gamma
        self.recent_index += 1

        if self.recent_index <= self.recent_num:
            self.recent_alpha = np.mean(self.recent_alpha_list[:self.recent_index])
            self.recent_gamma = np.mean(self.recent_gamma_list[:self.recent_index])
        else:
            self.recent_alpha = np.mean(self.recent_alpha_list)
            self.recent_gamma = np.mean(self.recent_gamma_list)


        print("avg sample reward {:.3f}, avg greedy reward {:.3f} recent_gamma {:.3f}".format(reward_sample_avg, reward_greedy_avg, self.recent_gamma))

        if self.mask_type == 0:
            # batch_size * seq_length
            mask = (sample_seq > 0).float()
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            mask = Variable(mask)
        elif self.mask_type == 1:
            # batch_size * seq_length
            mask = (greedy_seq > 0).float()
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            mask = Variable(mask)

        reward_diff = Variable(torch.from_numpy(reward_diff).float().cuda(), requires_grad=False)

        if self.use_gamma:
            temp_gamma = self.recent_gamma * self.gamma
        else:
            temp_gamma = 1

        # (batch_size * (seq_length + 1))
        output = - sample_seqLogprobs * reward_diff * mask * temp_gamma

        # seqLogprobs.reinforce(reward_diff)
        # output = - seqLogprobs * mask

        # average loss
        loss = output.sum() / mask.sum()

        # while loss.data[0] > 1 or loss.data[0] < -1:
        #     loss = loss * 0.1

        return loss, reward_diff.mean().data[0], sample_mean, greedy_mean

# RewardCriterion
# self-critical
class RewardMulOutCriterion(nn.Module):
    def __init__(self, opt):
        super(RewardMulOutCriterion, self).__init__()
        self.opt = opt

        self.reward_sample_total = 0
        self.reward_greedy_total = 0
        self.reward_num = 0

        self.alpha_type = opt.rl_alpha_type
        self.alpha = opt.rl_alpha_start
        self.recent_alpha = opt.rl_alpha_recent_start
        self.recent_num = opt.rl_alpha_recent_num

        self.recent_alpha_list = np.linspace(0, 0, self.recent_num)
        self.recent_gamma_list = np.linspace(0, 0, self.recent_num)
        self.recent_index = 0

        self.beta = opt.rl_beta
        self.gamma = opt.rl_gamma
        self.use_gamma = opt.rl_use_gamma

    # sample_seq    batch_size * seq_length
    # seqLogprobs   batch_size * seq_length
    # seq1          batch_size * seq_length
    # seqLogprobs1  batch_size * seq_length
    # target        batch_size * (seq_length + 1)  (the last is zero)
    # sample_seqLogprobs batch_size * seq_length * (vocab_size + 1)
    def forward(self, list_sample_seq, list_sample_seqLogprobs, greedy_seq, gts):

        all_loss = []
        all_reward_diff_mean = []
        all_sample_mean = []
        all_greedy_mean = []

        for i in range(len(list_sample_seq)):

            sample_seq = list_sample_seq[i]
            sample_seqLogprobs = list_sample_seqLogprobs[i]

            batch_size = sample_seq.size(0)
            seq_length = sample_seq.size(1)

            for i in range(batch_size):
                k = 0
                for j in range(seq_length):
                    if sample_seq[i, j] == 0:
                        k = 1
                    if k == 1:
                        sample_seq[i, j] = 0

            print("alpha {:.3f}, recent_alpha {:.3f}".format(self.alpha, self.recent_alpha))

            # greedy_seq : batch_size * seq_length
            # sample_seq : batch_size * seq_length
            # reward_diff : batch_size * seq_length

            if self.alpha_type == 0:
                temp_alpha = 1.0
            elif self.alpha_type == 1:
                temp_alpha = self.recent_alpha * self.beta
            elif self.alpha_type == 2:
                temp_alpha = self.alpha * self.beta

            reward_diff, sample_mean, greedy_mean = utils.get_self_critical_reward(greedy_seq, sample_seq, gts,
                                                                                   temp_alpha, self.opt)

            self.reward_sample_total += sample_mean
            self.reward_greedy_total += greedy_mean
            self.reward_num += 1

            reward_sample_avg = self.reward_sample_total / self.reward_num
            reward_greedy_avg = self.reward_greedy_total / self.reward_num

            self.alpha = self.reward_sample_total / self.reward_greedy_total

            # recent num
            self.recent_alpha_list[self.recent_index % self.recent_num] = sample_mean / greedy_mean
            if sample_mean - greedy_mean * temp_alpha == 0:
                temp_gamma = 1
            else:
                temp_gamma = 1 / np.abs(sample_mean - temp_alpha * greedy_mean)

            self.recent_gamma_list[self.recent_index % self.recent_num] = temp_gamma
            self.recent_index += 1

            if self.recent_index <= self.recent_num:
                self.recent_alpha = np.mean(self.recent_alpha_list[:self.recent_index])
                self.recent_gamma = np.mean(self.recent_gamma_list[:self.recent_index])
            else:
                self.recent_alpha = np.mean(self.recent_alpha_list)
                self.recent_gamma = np.mean(self.recent_gamma_list)

            print("avg sample reward {:.3f}, avg greedy reward {:.3f} recent_gamma {:.3f}".format(reward_sample_avg,
                                                                                                  reward_greedy_avg,
                                                                                                  self.recent_gamma))

            # batch_size * seq_length
            mask = (sample_seq > 0).float()

            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

            reward_diff = Variable(torch.from_numpy(reward_diff).float().cuda(), requires_grad=False)

            if self.use_gamma:
                temp_gamma = self.recent_gamma * self.gamma
            else:
                temp_gamma = 1

            # (batch_size * (seq_length + 1))
            output = - sample_seqLogprobs * reward_diff * Variable(mask) * temp_gamma

            # seqLogprobs.reinforce(reward_diff)
            # output = - seqLogprobs * mask

            # average loss
            loss = output.sum() / mask.sum()

            # while loss.data[0] > 1 or loss.data[0] < -1:
            #     loss = loss * 0.1

            all_loss.append(loss)
            all_reward_diff_mean.append(reward_diff.mean().data[0])
            all_sample_mean.append(sample_mean)
            all_greedy_mean.append(greedy_mean)

        final_loss = torch.cat([_ for _ in all_loss]).mean()
        final_reward_diff_mean = np.array(all_reward_diff_mean).mean()
        final_sample_mean = np.array(all_sample_mean).mean()
        final_greedy_mean = np.array(all_greedy_mean).mean()

        return final_loss, final_reward_diff_mean, final_sample_mean, final_greedy_mean

# # RewardCriterion
# # self-critical
# class RewardCriterion(nn.Module):
#     def __init__(self):
#         super(RewardCriterion, self).__init__()
#
#     # seq           batch_size * seq_length
#     # seqLogprobs   batch_size * seq_length
#     # seq1          batch_size * seq_length
#     # seqLogprobs1  batch_size * seq_length
#     # target        batch_size * (seq_length + 1)  (the last is zero)
#     def forward(self, seq, seqLogprobs, seq1, target, vocab):
#         # compute reward
#         # truncate to the same size
#         seq_length = seq.size(1)
#
#         # label
#         sent_label = utils.decode_sequence(vocab, target.data)
#
#         # sample
#         sent_seq = utils.decode_sequence(vocab, seq)
#
#         # greedy
#         sent_seq1 = utils.decode_sequence(vocab, seq1)
#
#         # train
#         # sample
#         reward, rewards = utils.get_reward(sent_seq, sent_label, "CIDEr")
#
#         # test
#         # greedy
#         reward1, rewards1 = utils.get_reward(sent_seq1, sent_label, "CIDEr")
#
#         # batch_size
#         reward_diff = rewards - rewards1
#         reward_diff = torch.from_numpy(reward_diff).float().cuda()
#         reward_diff = reward_diff.view(reward_diff.size(0), 1)
#
#         # repeat reward
#         # batch_size * seq_length
#         reward_diff = reward_diff.repeat(1, seq_length)
#
#         mask = (seq > 0).float()
#
#         # (batch_size * (seq_length + 1))
#         output = - seqLogprobs * Variable(reward_diff) * Variable(mask)
#
#         # seqLogprobs.reinforce(reward_diff)
#         # output = - seqLogprobs * mask
#
#         # average loss
#         loss = output.sum() / mask.sum()
#
#         return loss, reward_diff.mean()


# PolicyGradientCriterion
class PGCriterion:
    def __init__(self, opt):
        self.cureward_gamma = opt.cureward_gamma
        self.reward_base = opt.reward_base
        self.reward_gamma = opt.reward_gamma

    # input  : batch_size * (seq_length + 1) * (vocab_size + 1)
    # target : batch_size * (seq_length + 1)  (the last is zero)
    def forward_backward(self, input, target, mask, vocab):

        # compute reward
        batch_size, L, Mp1 = input.size(0), input.size(1), input.size(2)
        model_rewards = []
        model_probs = []
        sents = []
        sent_ts = []
        for b in range(batch_size):
            seq = []
            for l in range(L):
                prob = input[b, l]
                prob_prev = torch.exp(prob)
                it = prob_prev.multinomial()
                model_probs.append(it)
                seq.append(it.data[0])
            sent = utils.decode_sequence1(vocab, seq)

            seq_t = target[b].data
            sent_t = utils.decode_sequence2(vocab, seq_t)

            sents.append(sent)
            sent_ts.append(sent_t)

        reward_mean, sent_rewards = utils.get_reward(sents, sent_ts, "CIDEr")
        for b in range(batch_size):
            for l in range(L):
                if l != L - 1:
                    model_rewards.append(0)
            sent_reward = sent_rewards[b]
            sent_reward = (sent_reward - self.reward_base) * self.reward_gamma
            model_rewards.append(sent_reward)


        R = 0
        rewards = []
        for r in model_rewards[::-1]:
            if r != 0:
                R = 0
            R = r + self.cureward_gamma * R
            rewards.insert(0, R)

        rewards = torch.FloatTensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for prob, r in zip(model_probs, rewards):
            prob.reinforce(r)
        grad_variables = [None for _ in model_probs]
        autograd.backward(model_probs, grad_variables)

        # compute loss
        target = target[:,:input.size(1)].contiguous().data
        mask = mask[:, :input.size(1)].contiguous().data
        # (batch_size * (seq_length + 1)) * (vocab_size + 1)
        input = input.view(-1, input.size(2)).data
        # (batch_size * (seq_length + 1)) * 1
        target = target.view(-1, 1)
        # (batch_size * (seq_length + 1))
        output = - input.gather(1, target) * mask
        # average loss
        loss = output.sum() / mask.sum()
        return loss, reward_mean

# SCST
class SCSTCriterion(nn.Module):
    def __init__(self):
        super(SCSTCriterion, self).__init__()

    # input train
    # input1 test
    # target
    # train : seq
    # test : seq
    def forward(self, input, input1, seq, seq1, target, vocab):
        # truncate to the same size
        # input (batch_size * (seq_length + 2) * (vocab_size + 1))
        # target (batch_size * (seq_length))
        batch_size, L, Mp1 = input.size(0), input.size(1), input.size(2)
        seq_length = target.size(1)

        loss = Variable(torch.FloatTensor(1).zero_(),requires_grad=True).cuda()
        n = 0

        label = utils.decode_sequence(vocab, target.data)
        seq = utils.decode_sequence(vocab, seq)
        seq1 = utils.decode_sequence(vocab, seq1)

        # train
        reward = utils.get_reward(seq, label, "CIDEr")
        # test
        reward1 = utils.get_reward(seq1, label, "CIDEr")

        reward_diff = reward - reward1

        if reward_diff < 1:
            reward_diff = 1;

        for b in range(batch_size):
            first_time = True

            for t in range(1, L):

                if t - 1 >= seq_length:
                    target_index = 0
                else:
                    target_index = target.data[b, t-1]

                if target_index == 0 and first_time:
                    first_time = False
                elif target_index == 0 and not first_time:
                    break

                logsoft = input[b, t, target_index]
                loss.sub_(logsoft)
                n += 1

        loss.div_(n)
        loss.mul_(reward_diff)

        return loss, reward, reward1

