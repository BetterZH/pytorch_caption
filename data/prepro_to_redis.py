import os
import json
import argparse
import numpy as np
from scipy.misc import imread
import io
import redis
from tqdm import tqdm
import lmdb

def main(opt):
    map_size = int(1e12)
    env = lmdb.open(opt.lmdb_path, map_size=map_size)
    txn = env.begin(write=True)

    # r = redis.Redis(host=opt.redis_host, port=6379, db=0)

    print('prepro feature loading json file: ', opt.input_json)
    info = json.load(open(opt.input_json))
    images_root = opt.images_root

    len_images = len(info['images'])

    for i in tqdm(range(len_images)):

        img_info = info['images'][i]
        id = img_info['id']
        file_path = img_info['file_path']

        full_path = os.path.join(images_root, file_path)

        I = imread(full_path)

        output = io.BytesIO()
        np.savez(output, img=I)
        content = output.getvalue()

        # r.set(file_path, content)
        txn.put(str(file_path), content)

    txn.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='./dataset/caption_datasets/data_coco.json', help='')
    parser.add_argument('--images_root', type=str, default='./dataset', help='')
    parser.add_argument('--redis_host', type=str, default='127.0.0.1', help='')
    parser.add_argument('--lmdb_path', type=str, default='mscoco', help='')


    opt = parser.parse_args()
    print 'parsed input parameters:'
    print json.dumps(vars(opt), indent=2)

    main(opt)