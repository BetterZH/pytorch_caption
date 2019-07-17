# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import csv
import xlrd

path_xls = '/Users/zhuxinxin/我的坚果云/paper/paper_1/result/resnext64/da_nic_new.xlsx'

data = xlrd.open_workbook(path_xls)

table = data.sheets()[2]

ylabels = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr', 'ROUGE_L']

for k in range(6):

    x1 = []
    y1 = []

    x2 = []
    y2 = []

    is_head = True

    for i in range(31):
        row = i + 1
        col = k * 4
        if is_head:
            is_head = False
            label_1 = table.cell(row, col + 1).value
            label_2 = table.cell(row, col + 2).value
        else:
            val = table.cell(row, 0).value

            x1.append(float(val))
            x2.append(float(table.cell(row, col + 0).value))

            y1.append(float(table.cell(row, col + 1).value))
            y2.append(float(table.cell(row, col + 2).value))

    x_min = min(min(x1),min(x2))
    x_max = max(max(x1),max(x2))

    y_min = min(min(y1),min(y2))
    y_max = max(max(y1),max(y2))

    plt.plot(x1, y1, 'ro-', label=label_1)
    plt.plot(x2, y2, 'b*-', label=label_2)

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    plt.xlabel('iteration', fontsize=20)
    plt.ylabel(ylabels[k], fontsize=20)

    print(x_min, x_max)
    print(y_min, y_max)

    plt.xlim(x_min - 2500, x_max + 2500)
    plt.ylim(y_min - 0.01 , y_max + 0.01)

    # upper left lower right
    plt.legend(loc='lower right')

    plt.grid(True)

    # plt.show()

    plt.savefig('f' + str(k+5) + '.png')

    plt.close('all')

    print('save fig ' + str(k))