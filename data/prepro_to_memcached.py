import os
import json
import argparse
import numpy as np
from scipy.misc import imread, imresize
import io
import memcache
from tqdm import tqdm

def main(opt):

    mc = memcache.Client([opt.host])

    print('prepro feature loading json file: ', opt.input_json)
    info = json.load(open(opt.input_json))
    images_root = opt.images_root

    len_images = len(info['images'])

    for i in tqdm(range(len_images)):

        img_info = info['images'][i]
        id = img_info['id']
        file_path = img_info['file_path']

        I = imread(os.path.join(images_root, file_path))
        I = imresize(I, (opt.image_size, opt.image_size))

        output = io.BytesIO()
        np.savez(output, img=I)
        content = output.getvalue()

        mc.set(str(file_path), content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='../dataset/caption_datasets/data_coco.json', help='')
    parser.add_argument('--images_root', type=str, default='../images', help='')
    parser.add_argument('--host', type=str, default='127.0.0.1:11211', help='')
    parser.add_argument('--image_size', type=int, default=224,
                        help='size of the rnn in number of hidden nodes in each layer')

    opt = parser.parse_args()
    print 'parsed input parameters:'
    print json.dumps(vars(opt), indent=2)

    main(opt)