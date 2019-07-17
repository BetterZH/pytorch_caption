import io
import numpy as np
import db.lmdb_data as lmdb_data
import cv2
from scipy.misc import imread, imresize
import Dic2Obj

path_lmdb = '/media/amds/disk/code_cap/bottom-up-attention/output/aic_val_resnet101_fasterRcnn_lmdb'
lmdb_bu = lmdb_data.lmdb_data(path_lmdb)
lmdb_bu.open_for_read()

image_path = '/media/amds/disk/dataset/aic/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/0003a0755539c426ecfc7ed79bc74aeea6be740b.jpg'
image_id = '0003a0755539c426ecfc7ed79bc74aeea6be740b.jpg'
value = lmdb_bu.get(image_id)
data = np.load(io.BytesIO(value))
data_x = data['x'].tolist()

boxes = data_x['boxes']
np_boxes = np.array(boxes)

img = imread(image_path)
# print(img.shape[0])

img_h = float(img.shape[0])
img_w = float(img.shape[1])

print(img_h, img_w)

box_size = np.array([img_w, img_h, img_w, img_h])
np_nboxes = np_boxes / box_size

# print(np_boxes)
# print(np_nboxes)

region_boxes = np_nboxes * np.array([7, 7, 7, 7])
region_boxes = region_boxes.round().astype(int)
# print(region_boxes)

# print(region_boxes.dtype)
# print(region_boxes.astype(int))

# cv2.imshow('image', img)
# cv2.waitKey(0)



# print(boxes)

# def test(x1):
#     x1.a = 10
#
# x = {}
# x['a'] = 20
#
# x1 = Dic2Obj.Dic2Obj(x)
# print(x1.a)
# test(x1)
# print(x1.a)


#
# img = cv2.imread(image_path)
# cv2.imshow('image', img)
# cv2.waitKey(0)
#

for box in region_boxes:

    # print(box)

    x_1 = int(box[0])
    y_1 = int(box[1])

    x_2 = int(box[2])
    y_2 = int(box[3])

    print(y_1, y_2, x_1, x_2)

#
#     crop_img = img[y_1:y_2, x_1:x_2]
#
#     cv2.imshow('image', crop_img)
#     cv2.waitKey(0)
#
# cv2.waitKey(0)