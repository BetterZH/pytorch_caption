import numpy as np
import json

path_pkl = "/media/amds/6332baea-87b9-4f13-bd90-1ef635cd81d2/caption/caption_checkpoint/triple_atten/coco_triple_atten_n3p9_infos.pkl"
filename = "history/coco_triple_atten_n3p9_infos.json"

infos = np.load(path_pkl)
val_result_history = infos["val_result_history"]

with open(filename, "w") as f:
    json.dump(val_result_history, f)