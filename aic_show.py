#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import os

path_file = '/media/amds/data3/caption/aic_submit/captions_caption_test_images_20170923_model_ensemble_4.json'
path_file_w = '/media/amds/data3/caption/aic_submit/captions_caption_test_images_20170923_model_ensemble_4.txt'


predictions = json.load(open(path_file, 'r'))

file = open(path_file_w, 'w')
for pred in predictions:
    print(pred['image_id'] + ' ' + pred['caption'])
    file.write(pred['image_id'] + ' ' + pred['caption'] + '\n')

file.close()
