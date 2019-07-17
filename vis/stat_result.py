# -*- coding:utf-8 -*-
from os import listdir
from os.path import isfile, join
import csv
import zmq
import json
import time

path_roots = ['/home/amds/caption/caption_result/transformer',
              '/home/amds/caption/caption_result/transformer_177',
              '/home/amds/caption/caption_result/transformer_148']
path_result = '/home/amds/caption/caption_result/transformer.csv'

# path_roots = ['/Users/zhuxinxin/cloud/result']
# path_result = 'transformer.csv'

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

def get_model_result(path_roots):

    lines = []
    lines.append(",iteration,CIDEr,ROUGE_L,BLEU_1,BLEU_2,BLEU_3,BLEU_4\n")

    for path_root in path_roots:
        all_files = list_all_csv_files(path_root)
        for filename in all_files:
            model_name = get_model_name(filename)
            max_row = get_best_result(path_root, filename)
            # print(model_name)
            if max_row is not None:
                split = ","
                if len(max_row) == 6:
                    split = ",,"
                row = model_name + split + ",".join(max_row) + "\n"
                lines.append(row)

    # rows[1:].sort(key=lambda row: row.split(",")[0])
    # lines[1:] = sorted(lines[1:], cmp_ignore_case)

    # for line in lines:
    #     print(line.split(",")[0])
    # print("=====================")

    # with open(path_write, 'w') as f:
    #     f.writelines(lines)

    return "".join(lines)

class TransClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)

    def start(self, host = 'ali.deepai.pro', port = 5559):
        self.host = host
        self.port = port
        self.server_info = "tcp://" + self.host + ":" + str(self.port)
        self.socket.connect(self.server_info)
        print("connected to server: " + self.server_info)

    def reconnect(self):
        self.socket.close()
        self.socket.connect(self.server_info)
        print("reconnected to server: " + self.server_info)

    def close(self):
        self.socket.close()

    def send(self, msg):
        try:
            self.socket.send(msg)
        except:
            self.reconnect()


def write_model_result(results, path_result):
    with open(path_result, "w") as f:
        f.write(results)

client = TransClient()
client.start()

while True:
    try:
        dict_result = {}
        dict_result['type'] = 'result'
        results = get_model_result(path_roots)
        dict_result['data'] = results
        json_result = json.dumps(dict_result)
        client.send(json_result)
        cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        write_model_result(results, path_result)
        print("send success " + cur_time)
        time.sleep(1)
    except KeyboardInterrupt:
        print("exit")
        exit(0)

