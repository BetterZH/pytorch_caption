import json
import os

path_root = '/home/amds/caption/caption_aic/aicBleu/'

files = []
for file in os.listdir(path_root):
    files.append(file)
# files = ['captions_caption_test1_images_20170923_aicBleu4_0_results.json',
#          'captions_caption_test1_images_20170923_aicBleu4_7500_results.json',
#          'captions_caption_test1_images_20170923_aicBleu4_15000_results.json',
#          'captions_caption_test1_images_20170923_aicBleu4_22500_results.json']

predictions = []
keys = {}
for file in files:
    path_file = os.path.join(path_root, file)
    print(path_file)
    p = json.load(open(path_file, 'r'))

    for pred in p:
        if not keys.has_key(pred['image_id']):
            keys[pred['image_id']] = 1
            predictions.append(pred)

print(len(predictions))

path_result = os.path.join(path_root, 'captions_caption_test1_images_20170923_aicBleu4.json')
json.dump(predictions, open(path_result, 'w'))
