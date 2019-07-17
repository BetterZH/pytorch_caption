import os
import json

# sudo apt-get install axel
if not os.path.exists('annotations'):
    os.system('axel -n 100  http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip')
    os.system('unzip captions_train-val2014.zip')

val = json.load(open('annotations/captions_val2014.json', 'r'))
train = json.load(open('annotations/captions_train2014.json', 'r'))

print val.keys()
print val['info']
print len(val['images'])
print len(val['annotations'])
print val['images'][0]
print val['annotations'][0]

imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)

out = []
for i,img in enumerate(imgs):
    imgid = img['id']

    loc = 'train2014' if 'train' in img['file_name'] else 'val2014'

    jimg = {}
    jimg['file_path'] = os.path.join(loc, img['file_name'])
    jimg['id'] = imgid

    sents = []
    annotsi = itoa[imgid]
    for a in annotsi:
        sents.append(a['caption'])
    jimg['captions'] = sents
    out.append(jimg)

json.dump(out, open('coco_raw.json', 'w'))

