import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

path_json = '/home/amds/caption/caption_result/aic/aic_weight_1.json'
path_txt = '/home/amds/caption/caption_result/aic/aic_weight_1.txt'

contents = json.load(open(path_json))

with open(path_txt,'w') as f:
    for content in contents:
        txt = str(content['image_id']) + ' ' + ''.join(content['caption'].split())
        print(txt)
        f.write(txt  + '\n')