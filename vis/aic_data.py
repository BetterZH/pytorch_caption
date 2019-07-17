import json
import pymongo
from tqdm import tqdm
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

path_dataset = '/media/amds/data2/dataset/aic/dataset_aic.json'
path_dataset_txt = '/media/amds/data2/dataset/aic/dataset_aic.txt'

client = pymongo.MongoClient('localhost', 27017)
db = client['aic']
collection = db['data']

contents = json.load(open(path_dataset))

with open(path_dataset_txt,'w') as f:
    for i in tqdm(range(len(contents))):
        content = contents[i]
        # print(content)
        sents = []
        for sent in content['sentences']:
            sents.append(sent['raw'])

        imgid = content['imgid']
        filename = content['filename']
        sent = ','.join(sents)

        txt = str(imgid) + ' ' + filename + ' ' + sent
        # print(txt)
        f.write(txt + '\n')

        # collection.save(content)


