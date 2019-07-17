import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import vis.visual_conv as visual_conv
import common_utils
import numpy as np

# Model	Version	Prec@1	Prec@5
# InceptionResNetV2	Tensorflow	80.4	95.3
# InceptionV4	Tensorflow	80.2	95.3
# InceptionResNetV2	Our porting	80.170	95.234
# InceptionV4	Our porting	80.062	94.926
# ResNeXt101_64x4d	Torch7	79.6	94.7
# ResNeXt101_64x4d	Our porting	78.956	94.252
# ResNeXt101_32x4d	Torch7	78.8	94.4
# ResNet152	Pytorch	78.312	94.046
# ResNeXt101_32x4d	Our porting	78.188	93.886
# ResNet152	Torch7	77.84	93.84
# ResNet152	Our porting	77.386	93.594

# Resnet 12L, 2048L, 7L, 7L
# Inception 12L, 1536L, 8L, 8L


# with no linear
# class resnet_att_mul_linear(nn.Module):
#     def __init__(self, resnet, opt):
#         super(resnet_att_mul_linear, self).__init__()
#
#         self.att_size = opt.att_size
#
#         self.resnet = resnet
#
#         self.conv1 = nn.Conv1d(self.att_size, self.att_size, 2, 2)
#
#
#     def forward(self, x):
#
#         # visual_conv.vis_conv(x.data.cpu().numpy(),0)
#
#         #   2048 * 7 * 7
#         # x batch_size * channels * w *h
#         for i in range(7):
#             x = self.resnet[i](x)
#             # print(x.size())
#             if i == 5:
#                 x_5 = x
#             elif i == 6:
#                 x_6 = x
#
#         # 512 * 28 * 28 -> 512 * 14 * 14 -> 512 * (14 * 14) -> (14 * 14) * 512
#         x_5_feats = F.avg_pool2d(x_5, 2).view(x_5.size(0), x_5.size(1), -1).transpose(1, 2).contiguous()
#
#         # 1024 * 14 * 14 -> 1024 * (14 * 14) -> (14 * 14) * 1024
#         x_6_feats = x_6.view(x.size(0), x.size(1), -1).transpose(1, 2).contiguous()
#
#         # (14 * 14) * 1024 -> (14 * 14) * 512
#         x_6_feats = self.conv1(x_6_feats)
#
#         # (14 * 14) * 512 -- 196 * 512
#         feats = x_5_feats + x_6_feats
#
#         # visual_conv.vis_conv(x_5.data.cpu().numpy(),5)
#         # visual_conv.vis_conv(x_6.data.cpu().numpy(),6)
#         # visual_conv.vis_conv(feats.transpose(1, 2).contiguous().view(feats.size(0),512,14,14).data.cpu().numpy(), 7)
#
#         return feats


# resnet_att
class resnet_att(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_att, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size
        self.adaptive_size = opt.adaptive_size

        self.resnet = resnet

    def forward(self, x):
        #   2048 * 7 * 7
        # x batch_size * channels * w *h
        x = self.resnet(x)

        # batch_size * channels * adaptive_size * adaptive_size
        x = F.adaptive_avg_pool2d(x, self.adaptive_size)

        # batch_size * channels
        # batch_size * feat_size
        # fc_feats = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))

        # batch_size * (w*h) * channels
        # batch_size * att_size * att_feat_size
        att_feats = x.view(x.size(0), x.size(1), -1).transpose(1, 2).contiguous()


        # print(fc.size())
        # print(att.size())

        # self.resnet.remove()

        return att_feats


# with no linear
# class resnet_att(nn.Module):
#     def __init__(self, resnet, opt):
#         super(resnet_att, self).__init__()
#
#         self.fc_feat_size = opt.fc_feat_size
#         self.att_feat_size = opt.att_feat_size
#         self.att_size = opt.att_size
#         self.rnn_size = opt.rnn_size
#         self.pool_size = opt.pool_size
#
#         self.resnet = resnet
#
#     def forward(self, x):
#         #   2048 * 7 * 7
#         # x batch_size * channels * w *h
#         x = self.resnet(x)
#
#         # batch_size * channels
#         # batch_size * feat_size
#         # fc_feats = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))
#
#         # batch_size * (w*h) * channels
#         # batch_size * att_size * att_feat_size
#         att_feats = x.view(x.size(0), x.size(1), -1).transpose(1, 2).contiguous()
#
#         # att_feats = F.adaptive_avg_pool2d(att_feats, 4)
#
#         # print(fc.size())
#         # print(att.size())
#
#         # self.resnet.remove()
#
#         return att_feats

# # with no linear
# class resnet_fc_att(nn.Module):
#     def __init__(self, resnet, opt):
#         super(resnet_fc_att, self).__init__()
#
#         self.fc_feat_size = opt.fc_feat_size
#         self.att_feat_size = opt.att_feat_size
#         self.att_size = opt.att_size
#         self.rnn_size = opt.rnn_size
#         self.pool_size = opt.pool_size
#
#         self.resnet = resnet
#
#
#     def forward(self, x):
#
#         #   2048 * 7 * 7
#         # x batch_size * channels * w *h
#         x = self.resnet(x)
#
#         # batch_size * channels
#         # batch_size * feat_size
#         fc_feats = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))
#
#         # batch_size * (w*h) * channels
#         # batch_size * att_size * att_feat_size
#         att_feats = x.view(x.size(0), x.size(1), -1).transpose(1,2).contiguous()
#
#         # print(fc.size())
#         # print(att.size())
#
#         # self.resnet.remove()
#
#         return fc_feats, att_feats

# with no linear
class resnet_fc_att(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_fc_att, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size
        self.att_pool_size = getattr(opt, 'att_pool_size', 1)
        self.use_pre_feat = getattr(opt, 'use_pre_feat', False)
        self.adaptive_size = getattr(opt, 'adaptive_size', 7)

        self.resnet = resnet

    def forward(self, x):
        #   2048 * 7 * 7
        # x: batch_size * channels * h * w
        if not self.use_pre_feat:
            x = self.resnet(x)
        h = x.size(2)
        w = x.size(3)

        # batch_size * channels
        # batch_size * feat_size
        fc_feats = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))

        # batch_size * (w*h) * channels
        # batch_size * att_size * att_feat_size
        if self.adaptive_size != h and self.adaptive_size != w:
            x = F.adaptive_avg_pool2d(x, self.adaptive_size)
        att_feats = F.avg_pool2d(x, self.att_pool_size).view(x.size(0), x.size(1), -1).transpose(1, 2).contiguous()

        # print(fc_feats.size())
        # print(att_feats.size())

        # self.resnet.remove()

        return fc_feats, att_feats

# with linear
class resnet_fc_att_linear(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_fc_att_linear, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size


        self.resnet = resnet
        self.linear1 = nn.Linear(self.fc_feat_size, self.rnn_size)
        self.linear2 = nn.Linear(self.att_feat_size, self.rnn_size)

        self.relu = nn.PReLU()

        init.xavier_normal(self.linear1.weight)
        init.xavier_normal(self.linear2.weight)


    def forward(self, x):

        #   2048 * 7 * 7
        # x batch_size * channels * w *h
        x = self.resnet(x)

        # batch_size * channels
        # batch_size * feat_size
        fc = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))

        # batch_size * rnn_size
        fc_feats = self.linear1(fc)
        fc_feats = self.relu(fc_feats)

        # batch_size * (w*h) * channels
        # batch_size * att_size * att_feat_size
        att = x.view(x.size(0), x.size(1), -1).transpose(1,2).contiguous()
        # (batch_size * att_size) * att_feat_size
        att_feats = self.linear2(att.view(-1, self.att_feat_size))
        # batch_size * att_size * att_feat_size
        att_feats = att_feats.view(-1, self.att_size, self.rnn_size)
        att_feats = self.relu(att_feats)

        # print(fc.size())
        # print(att.size())

        # self.resnet.remove()

        return fc_feats, att_feats



class resnet_fc(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_fc, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size


        self.resnet = resnet


    def forward(self, x):

        #   2048 * 7 * 7
        # x batch_size * channels * w *h
        x = self.resnet(x)

        # vis
        # visual_conv.vis_conv(x.data.cpu().numpy())

        # batch_size * channels
        # batch_size * feat_size
        fc_feats = F.avg_pool2d(x, self.pool_size).squeeze()

        # batch_size * rnn_size

        # print(fc.size())

        return fc_feats

class resnet_fc_linear(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_fc_linear, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size


        self.resnet = resnet
        self.linear1 = nn.Linear(self.fc_feat_size, self.rnn_size)

        self.relu = nn.PReLU()

        init.xavier_normal(self.linear1.weight)

    def forward(self, x):
        #   2048 * 7 * 7
        # x batch_size * channels * w *h
        x = self.resnet(x)

        # batch_size * channels
        # batch_size * feat_size
        fc = F.avg_pool2d(x, self.pool_size).squeeze()

        # batch_size * rnn_size
        fc_feats = self.linear1(fc)
        fc_feats = self.relu(fc_feats)

        # print(fc.size())

        return fc_feats

# with no linear
class resnet_with_regions(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_with_regions, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size
        self.att_pool_size = getattr(opt, 'att_pool_size', 1)
        self.use_pre_feat = getattr(opt, 'use_pre_feat', False)

        self.resnet = resnet

    def forward(self, x):
        #   2048 * 7 * 7
        # x batch_size * channels * w *h
        if not self.use_pre_feat:
            x = self.resnet(x)

        # batch_size * channels
        # batch_size * feat_size
        fc_all_feats = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))

        batch_size = fc_all_feats.size(0) / (self.att_size + 1)

        list_fc_feats = []
        list_att_feats = []

        for i in range(batch_size):
            list_fc_feats.append(fc_all_feats[i*(self.att_size+1)])
            list_att_feats.append(fc_all_feats[i*(self.att_size+1)+1: (i+1)*(self.att_size+1)])

        fc_feats = torch.cat(list_fc_feats, 0).view(batch_size, self.fc_feat_size)
        att_feats = torch.cat(list_att_feats, 0).view(batch_size, self.att_size, self.att_feat_size)

        return fc_feats, att_feats

# with no linear
class resnet_with_conv_regions(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_with_conv_regions, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_size = opt.att_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size
        self.att_pool_size = getattr(opt, 'att_pool_size', 1)
        self.use_pre_feat = getattr(opt, 'use_pre_feat', False)

        self.resnet = resnet

    def forward(self, input):

        images = input['images']
        np_nboxes = input['boxes']

        # 2048 * 7 * 7
        # 224 * 224 -> 7 * 7
        # x batch_size * channels * h * w
        if not self.use_pre_feat:
            x = self.resnet(images)

        batch_size = x.size(0)

        # batch_size * channels
        # batch_size * feat_size
        fc_feats = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))

        box_size = np.array([self.pool_size, self.pool_size, self.pool_size, self.pool_size])
        region_boxes = np_nboxes * box_size
        region_boxes = region_boxes.round().astype(int)

        list_att_feats = []
        for i in range(batch_size):
            batch_regions = region_boxes[i]
            list_batch_att_feats = []
            for j in range(self.att_size):
                region = batch_regions[j]

                x_1 = int(region[0])
                y_1 = int(region[1])
                x_2 = int(region[2])
                y_2 = int(region[3])

                if x_1 == x_2:
                    if x_2 < self.pool_size:
                        x_2 += 1
                    elif x_1 > 0:
                        x_1 -= 1

                if y_1 == y_2:
                    if y_2 < self.pool_size:
                        y_2 += 1
                    elif y_1 > 0:
                        y_1 -= 1

                # (y_2-y_1) * (x_2-x_1)
                kernel_size = (y_2-y_1, x_2-x_1)

                # print(x_1, y_1, x_2, y_2)

                # 1 * att_feat_size * (y_2-y_1) * (x_2-x_1)
                att_region = x[i, :, y_1:y_2, x_1:x_2].contiguous().view(1, self.att_feat_size, y_2-y_1, x_2-x_1)

                # 1 * att_feat_size * 1 * 1
                att_pool_feat = F.avg_pool2d(att_region, kernel_size)

                # 1 * att_feat_size
                att_feat = att_pool_feat.view(1, self.att_feat_size)
                list_batch_att_feats.append(att_feat)

            # att_size * att_feat_size
            batch_att_feats = torch.cat(list_batch_att_feats, 0).view(self.att_size, self.att_feat_size)
            list_att_feats.append(batch_att_feats)

        # batch_size * att_size * att_feat_size
        att_feats = torch.cat([_.unsqueeze(0) for _ in list_att_feats], 0).view(batch_size, self.att_size, self.att_feat_size)

        return fc_feats, att_feats


# with no linear
class resnet_with_att_conv_regions(nn.Module):
    def __init__(self, resnet, opt):
        super(resnet_with_att_conv_regions, self).__init__()

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.bu_feat_size = opt.bu_feat_size
        self.att_size = opt.att_size
        self.bu_size = opt.bu_size
        self.rnn_size = opt.rnn_size
        self.pool_size = opt.pool_size
        self.att_pool_size = getattr(opt, 'att_pool_size', 1)
        self.use_pre_feat = getattr(opt, 'use_pre_feat', False)

        self.resnet = resnet

    def forward(self, input):

        images = input['images']
        np_nboxes = input['boxes']

        # 2048 * 7 * 7
        # 224 * 224 -> 7 * 7
        # x batch_size * channels * h * w
        if not self.use_pre_feat:
            x = self.resnet(images)

        batch_size = x.size(0)

        # batch_size * channels
        # batch_size * feat_size
        fc_feats = F.avg_pool2d(x, self.pool_size).view(x.size(0), x.size(1))

        # batch_size * (w*h) * channels
        # batch_size * att_size * att_feat_size
        att_feats = F.avg_pool2d(x, self.att_pool_size).view(x.size(0), x.size(1), -1).transpose(1, 2).contiguous()

        # bu_featus
        box_size = np.array([self.pool_size, self.pool_size, self.pool_size, self.pool_size])
        region_boxes = np_nboxes * box_size
        region_boxes = region_boxes.round().astype(int)

        list_bu_feats = []
        for i in range(batch_size):
            batch_regions = region_boxes[i]
            list_batch_bu_feats = []
            for j in range(self.bu_size):
                region = batch_regions[j]

                x_1 = int(region[0])
                y_1 = int(region[1])
                x_2 = int(region[2])
                y_2 = int(region[3])

                if x_1 == x_2:
                    if x_2 < self.pool_size:
                        x_2 += 1
                    elif x_1 > 0:
                        x_1 -= 1

                if y_1 == y_2:
                    if y_2 < self.pool_size:
                        y_2 += 1
                    elif y_1 > 0:
                        y_1 -= 1

                # (y_2-y_1) * (x_2-x_1)
                kernel_size = (y_2 - y_1, x_2 - x_1)

                # print(x_1, y_1, x_2, y_2)

                # 1 * bu_feat_size * (y_2-y_1) * (x_2-x_1)
                bu_region = x[i, :, y_1:y_2, x_1:x_2].contiguous().view(1, self.bu_feat_size, y_2 - y_1,
                                                                         x_2 - x_1)

                # 1 * bu_feat_size * 1 * 1
                bu_pool_feat = F.avg_pool2d(bu_region, kernel_size)

                # 1 * bu_feat_size
                bu_feat = bu_pool_feat.view(1, self.bu_feat_size)
                list_batch_bu_feats.append(bu_feat)

            # att_size * att_feat_size
            batch_att_feats = torch.cat(list_batch_bu_feats, 0).view(self.bu_size, self.bu_feat_size)
            list_bu_feats.append(batch_att_feats)

        # batch_size * att_size * att_feat_size
        bu_featus = torch.cat([_.unsqueeze(0) for _ in list_bu_feats], 0).view(batch_size, self.bu_size,
                                                                                self.bu_feat_size)
        # fc_feats  : batch_size * feat_size
        # att_feats : batch_size * att_size * att_feat_size
        # bu_featus : batch_size * att_size * att_feat_size
        return fc_feats, att_feats, bu_featus



# # with linear
# class resnet_fc_att_linears(nn.Module):
#     def __init__(self, resnet, opt):
#         super(resnet_fc_att_linears, self).__init__()
#
#         self.fc_feat_size = opt.fc_feat_size
#         self.att_feat_size = opt.att_feat_size
#         self.att_size = opt.att_size
#         self.rnn_size = opt.rnn_size
#         self.rnn_size_list = opt.rnn_size_list
#
#         self.resnet = resnet
#         self.linear1s = nn.ModuleList()
#         self.linear2s = nn.ModuleList()
#
#         for size in self.rnn_size_list:
#             linear1 = nn.Linear(self.fc_feat_size, size)
#             init.xavier_normal(linear1.weight)
#             self.linear1s.append(linear1)
#
#             linear2 = nn.Linear(self.att_feat_size, size)
#             init.xavier_normal(linear2.weight)
#             self.linear2s.append(linear2)
#
#         self.relu = nn.PReLU()
#
#     def forward(self, x):
#
#         #   2048 * 7 * 7
#         # x batch_size * channels * w *h
#         x = self.resnet(x)
#
#         # batch_size * channels
#         # batch_size * feat_size
#         fc = F.avg_pool2d(x, 7).squeeze()
#
#
#         fc_feats_list = []
#         att_feats_list = []
#         for i in range(len(self.rnn_size_list)):
#             # batch_size * rnn_size
#             fc_feats = self.linear1s[i](fc)
#             fc_feats = self.relu(fc_feats)
#             fc_feats_list.append(fc_feats)
#
#             # batch_size * (w*h) * channels
#             # batch_size * att_size * att_feat_size
#
#             att = x.view(x.size(0), x.size(1), -1).transpose(1,2).contiguous()
#             # (batch_size * att_size) * att_feat_size
#             att_feats = self.linear2s[i](att.view(-1, self.att_feat_size))
#             # batch_size * att_size * att_feat_size
#             att_feats = att_feats.view(-1, self.att_size, self.rnn_size_list[i])
#             att_feats = self.relu(att_feats)
#             att_feats_list.append(att_feats)
#
#         # print(fc.size())
#         # print(att.size())
#
#         # self.resnet.remove()
#
#         return fc_feats_list, att_feats_list