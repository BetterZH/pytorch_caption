import json
import os

path_result = 'captions_caption_test1_images_20170923_aicBleu4.json'

path_new = 'captions_caption_test1_images_20170923_aicBleu4.json'

path_old = 'captions_caption_test1_images_20170923_aicBleu4.json'

json_new = json.load(open(path_new, 'r'))

json_old = json.load(open(path_old, 'r'))


new_predict = {}
old_predict = {}

for pred in json_new:
    new_predict[pred['image_id']] = pred

all_keys = {}

predictions = []
for pred in json_old:
    if new_predict.has_key(pred['image_id']):
        predictions.append(new_predict[pred['image_id']])
    else:
        predictions.append(pred)

print(len(predictions))

json.dump(predictions, open(path_result, 'w'))