from pycrayon import CrayonClient
import time
import requests

class TensorBoard:
    def __init__(self):
        pass

    def start(self, id, host="127.0.0.1", port=8889):
        self.id = id
        self.host = host
        self.port = port
        self.cc = CrayonClient(hostname=self.host, port=self.port)
        names = self.cc.get_experiment_names()
        if id in names:
            try:
                self.exp = self.cc.open_experiment(id)
            except Exception, e:
                self.exp = self.cc.create_experiment(id)
        else:
            try:
                self.exp = self.cc.create_experiment(id)
            except Exception, e:
                self.exp = self.cc.open_experiment(id)

    def loss_train(self, val, step):
        self.exp.add_scalar_value("loss_train", val, step=step)

    def loss_val(self, val, step):
        self.exp.add_scalar_value("loss_val", val, step=step)

    def accuracy(self, val, step):
        self.exp.add_scalar_dict(val, step=step)

    def val(self, name, val, step):
        self.exp.add_scalar_value(name, val, step=step)

    def remove_all(self):
        self.cc = CrayonClient(hostname="127.0.0.1")
        self.cc.remove_all_experiments()

    def remove(self,host,id):
        self.cc = CrayonClient(hostname=host)
        self.cc.remove_experiment(id)

# board = TensorBoard()
# board.remove("127.0.0.1","transformer_more_sup_64a4n6")
