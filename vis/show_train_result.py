# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import sys
import csv
import collections
reload(sys)
sys.setdefaultencoding('utf-8')

# ids =  ['transformer_ms_o1_atten_64a4n6',
#         'transformer_ms_o2_atten_64a4n6',
#         'transformer_ms_o3_atten_64a4n6',
#         'transformer_ms_o4_atten_64a4n6',
#         'transformer_ms_o5_atten_64a4n6',
#         'transformer_ms_o6_atten_64a4n6',
#        ]

ids = ['transformer_ms_o6_atten_64a4n6_f_lr4e5_1e5']


path_root = '/Users/zhuxinxin/我的坚果云/paper/paper_12/result_csv/'

plt.xlabel('iteration', fontsize=20)
plt.ylabel('CIDEr', fontsize=20)
plt.grid(True)
plt.legend(loc='lower right')

for id in ids:

    path_csv = path_root + id + '.csv'

    results = collections.OrderedDict()

    with open(path_csv, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            # print(row[0],row[1])
            results[row[0]] = row[1]

    x1 = []
    y1 = []
    for k,v in results.items():
        print k, v
        x1.append(k)
        y1.append(v)

    plt.plot(x1, y1, label=id)

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    # upper left lower right


    # plt.savefig(id + '.png')
    #
    # plt.close('all')

plt.show()


