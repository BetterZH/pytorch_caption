import tensorboard
import json
import zmq
from notify.notify import notify

class TransServer:

    def setup(self):

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind('tcp://*:5558')

        self.boards = {}
        self.notifier = notify()
        self.notifier.login()

        self.board_host = "172.104.125.177"

        print('start server')
        while True:
            self.handle()

    def get_board(self, id):
        if not self.boards.has_key(id):
            board = tensorboard.TensorBoard()
            board.start(id, self.board_host)
            self.boards[id] = board
        return self.boards[id]

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.socket.recv()
        print self.data

        if len(self.data) > 0:
            data_dict = json.loads(self.data)
            print(data_dict)

            if data_dict['type'] == '0':

                id = data_dict['id']
                name = data_dict['name']
                val = data_dict['val']
                step = data_dict['step']

                board = self.get_board(id)
                board.val(name, val, step)

            elif data_dict['type'] == '1':

                id = data_dict['id']
                dict1 = data_dict['dict']
                step = data_dict['step']

                board = self.get_board(id)
                board.accuracy(dict1, step)

            elif data_dict['type'] == '2':

                id = data_dict['id']
                title = data_dict['title']
                msg = data_dict['msg']

                self.notifier.send(title, msg)


        # just send back the same data, but upper-cased

if __name__ == "__main__":
    server = TransServer()
    server.setup()