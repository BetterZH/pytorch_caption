# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import csv

path_csv = '/Users/zhuxinxin/我的坚果云/paper/paper_1/csv/da_nic_cider.csv'

x1 = []
y1 = []

x2 = []
y2 = []

is_head = True
with open(path_csv, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        if is_head:
            is_head = False
            label_1 = row[1]
            label_2 = row[2]
        else:
            x1.append(float(row[0]))
            x2.append(float(row[0]))

            y1.append(float(row[1]))
            y2.append(float(row[2]))

x_min = min(min(x1),min(x2))
x_max = max(max(x1),max(x2))

y_min = min(min(y1),min(y2))
y_max = max(max(y1),max(y2))

plt.plot(x1, y1, 'ro-', label=label_1)
plt.plot(x2, y2, 'g*-', label=label_2)

plt.xlabel('iteration')
plt.ylabel('CIDEr')

print(x_min, x_max)
print(y_min, y_max)

plt.xlim(x_min - 2500, x_max + 2500)
plt.ylim(y_min - 0.01 , y_max + 0.01)

plt.legend(loc='upper left')

plt.grid(True)

plt.show()