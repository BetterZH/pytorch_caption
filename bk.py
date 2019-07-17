import os
import shutil
import time

root = '/home/amds/test'
to = '/home/amds/to'
while True:
    for i in os.listdir(root):
        from_file = os.path.join(root,i)
        if os.path.isfile(from_file):
            to_file = os.path.join(to,i+"."+str(time.time()))
            shutil.move(from_file, to_file)
            print('move', from_file, to_file)
