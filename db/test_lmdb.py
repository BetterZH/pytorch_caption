import lmdb_data
import random
import time
import numpy as np
from scipy.misc import imread, imresize
import io
import math

lmdb1 = lmdb_data.lmdb_data("/home/amds/students",int(1e9))

# lmdb1.open_for_read()
# lmdb1.display()

print(int(math.ceil(10.0/3)))

lmdb1.open_for_write()
start = time.time()
for i in range(10):
    start_1 = time.time()
    I = imread("/home/amds/Desktop/COCO_test2014_000000000001.jpg")

    output = io.BytesIO()
    # np.savez(output, x=I) # 921788
    np.savez_compressed(output, x=I) # 726509
    np.save(output, I)
    content = output.getvalue()

    lmdb1.insert(str(i), content)

    data = np.load(io.BytesIO(content))
    x = data['x']

    # print(I==x)
    print("time {:.3f}".format(time.time() - start_1))

lmdb1.commit()
print("time {:.3f}".format(time.time()-start))

lmdb1.close()