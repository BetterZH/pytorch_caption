import json
import argparse
import zmq
import threading
import time
import os

class TransServer:

    def setup(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind('tcp://*:5559')
        print('started server')
        self.stats = {}

        self.thread = threading.Thread(target=self.run)
        self.thread.setDaemon(True)
        self.thread.start()

    def run(self):
        while True:
            self.handle()

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.socket.recv()
        dict_stat = json.loads(self.data)
        type = dict_stat['type']
        data = dict_stat['data']
        cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print("get data " + cur_time)
        if type == 'result':
            self.write_result_html(data, 0)
            self.write_result_html(data, 1)

    def cmp_ignore_case1(self, s1, s2):
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

    def cmp_ignore_case2(self, s1, s2):
        u1 = s1.split(",")[2]
        u2 = s2.split(",")[2]
        c1 = float(u1)
        c2 = float(u2)
        if c1 > c2:
            return -1
        elif c1 < c2:
            return 1
        return 0

    def write_result_html(self, data, cmp_type=0):
        html = '''
        <style type="text/css">
            table {border-collapse: collapse;}
            table,td {border: 1px solid black;}
            table td{padding: 3px;}
        </style>'''
        html += "<table>"
        rows = data.split("\n")[:-1]
        cmp_ignore_cases = []
        cmp_ignore_cases.append(self.cmp_ignore_case1)
        cmp_ignore_cases.append(self.cmp_ignore_case2)

        # rows[1:].sort(key=lambda row: row.split(",")[0])
        rows[1:] = sorted(rows[1:], cmp_ignore_cases[cmp_type])

        len_rows = len(rows)
        for i in range(len_rows):
            row = rows[i]
            if len(row) > 1:
                html += "<tr>"
                cols = row.split(",")
                len_cols = len(cols)
                for j in range(len_cols):
                    col = cols[j]
                    if i > 0 and j >= 2:
                        col = "{:.3f}".format(float(col))
                    html += "<td>" + col + "</td>"
                html += "</tr>"

        with open("result_" + str(cmp_type) + ".html", "w") as f:
            f.write(html)

if __name__ == "__main__":

    server = TransServer()
    server.setup()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("exit")
            exit(0)

