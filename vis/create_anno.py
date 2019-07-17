import json
from tqdm import tqdm

with open("anno_coco.json") as f:
    json_anno = json.load(f)

with open("triple_atten/coco_triple_atten_n3p9.json") as f:
    test_anno = json.load(f)

with open("data_coco.json") as f:
    data_coco = json.load(f)

test_keys = {}
for anno in test_anno:
    test_keys[anno["image_id"]] = ""
print(len(test_keys.keys()))

for data in data_coco["images"]:
    if test_keys.has_key(data["id"]):
        test_keys[data["id"]] = data["file_path"]

annotations = json_anno["annotations"]

dict_json = {}
for i in tqdm(range(len(annotations))):
    anno = annotations[i]
    # print(anno)
    image_id = anno["image_id"]
    if test_keys.has_key(image_id):
        if not dict_json.has_key(image_id):
            dict_json[image_id] = {"file_path": test_keys[image_id], "captions": []}
        dict_json[image_id]["captions"].append(anno["caption"].strip())

with open("dict_coco.json", "w") as f:
    json.dump(dict_json, f)
