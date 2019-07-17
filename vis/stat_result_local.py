# -*- coding:utf-8 -*-
from os import listdir
from os.path import isfile, join
import csv

path_roots = ['/Users/zhuxinxin/cloud/result']
path_result = '/Users/zhuxinxin/cloud/transformer.csv'

def list_all_csv_files(file_path):
    return [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith('.csv') and not f.startswith(".")]

def get_model_name(filename):
    model_anme = filename.split(".")[0]
    return model_anme

def get_best_result(path_root, filename):
    path_csv = join(path_root, filename)
    max_row = None
    try:
        with open(path_csv, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                if max_row is None:
                    max_row = row
                elif float(row[1]) >= float(max_row[1]):
                    max_row = row
    except:
        print("error: " + filename)
    return max_row

def cmp_ignore_case(s1, s2):
    u1 = s1.split(",")[0].lower()
    u2 = s2.split(",")[0].lower()
    len_u1 = len(u1)
    len_u2 = len(u2)
    min_len = min(len_u1, len_u2)
    for i in range(min_len):
        w1 = u1[i]
        w2 = u2[i]
        if w1 < w2:
            return -1
        elif w1 > w2:
            return 1
    if len_u1 < len_u2:
        return -1
    elif len_u1 > len_u2:
        return 1
    return 0

def cmp_ignore_case2(s1, s2):
    u1 = s1.split(",")[2]
    u2 = s2.split(",")[2]
    c1 = float(u1)
    c2 = float(u2)
    if c1 > c2:
        return -1
    elif c1 < c2:
        return 1
    return 0

def get_model_result(path_roots):

    lines = []
    # lines.append(",CIDEr,ROUGE_L,BLEU_1,BLEU_2,BLEU_3,BLEU_4,LOSS\n")
    lines.append(",iteration,CIDEr,ROUGE_L,BLEU_1,BLEU_2,BLEU_3,BLEU_4\n")


    for path_root in path_roots:
        all_files = list_all_csv_files(path_root)
        for filename in all_files:
            model_name = get_model_name(filename)
            max_row = get_best_result(path_root, filename)
            if max_row is not None:
                split = ","
                if len(max_row) == 6:
                    split = ",,"
                row = model_name + split + ",".join(max_row) + "\n"
                lines.append(row)
    lines[1:] = sorted(lines[1:], cmp_ignore_case2)

    return "".join(lines)


def write_model_result(results, path_result):
    with open(path_result, "w") as f:
        f.write(results)

results = get_model_result(path_roots)

write_model_result(results, path_result)



