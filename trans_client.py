from socket import *
import json
import time
import zmq

class TransClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)

    def start(self, id, host = 'localhost', port = 5558):
        self.id = id
        self.host = host
        self.port = port
        server_info = "tcp://" + self.host + ":" + str(self.port)
        print("server_info: " + server_info)
        self.socket.connect(server_info)

    def close(self):
        self.socket.close()

    def send(self,msg):
        self.socket.send('%s' % msg)

    def send_dict(self,msg_dict):
        msg = json.dumps(msg_dict)
        print(msg)
        self.socket.send('%s' % msg)

    def loss_train(self, val, step):
        msg_dict = {"id": self.id, "type":"0", "name": "loss_train", "val":val,"step":step}
        self.send_dict(msg_dict)

    def loss_val(self, val, step):
        msg_dict = {"id": self.id, "type":"0", "name": "loss_val", "val":val,"step":step}
        self.send_dict(msg_dict)

    def accuracy(self, val, step):
        msg_dict = {"id": self.id, "type": "1", "step": step}
        msg_dict["dict"] = val
        self.send_dict(msg_dict)

    def val(self, name, val, step):
        msg_dict = {"id": self.id, "type": "0", "name": name, "val": val, "step": step}
        self.send_dict(msg_dict)

    def notify_send(self, title, msg):
        msg_dict = {"id": self.id, "type": "2", "title": title, "msg": msg}
        self.send_dict(msg_dict)


if __name__ == "__main__":
    client = TransClient()
    client.start("test_id")

    for i in range(1000):
        client.val("loss_train", 0.1 + i * 0.001, 1)
        time.sleep(1)
        print('send')

