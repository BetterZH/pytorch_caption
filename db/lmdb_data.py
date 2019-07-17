import lmdb

class lmdb_data():

    def __init__(self, path, map_size=int(1e11)):
        self.path = path
        self.map_size = map_size

    def open_for_read(self):
        self.env = lmdb.open(self.path)
        self.txn = self.env.begin()

    def open_for_write(self):
        self.env = lmdb.open(self.path, map_size=self.map_size)
        self.txn = self.env.begin(write=True)

    def insert(self, key, value):
        self.txn.put(key, value)

    def commit(self):
        self.txn.commit()

    def get(self, key):
        return self.txn.get(key)

    def close(self):
        self.env.close()

    def display(self):
        cur = self.txn.cursor()
        for key, value in cur:
            print (key, value)
