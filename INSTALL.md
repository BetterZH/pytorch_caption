### install
pip install virtualenv
virtualenv pytorch --no-site-packages
source pytorch/bin/activate

### install pytroch

conda install pytorch torchvision cuda80 -c soumith

pip install pyyaml
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
pip install torchvision

sudo apt-get install python-tk
pip install requests
pip install --upgrade django==1.3
pip install h5py scipy pycrayon six matplotlib scikit-image jieba googletrans lmdb itchat jieba tqdm 
pip install image pymongo
pip install cffi
pip install pytorch_fft
pip install ipdb zmq
pip install opencv-python
pip install imguag

pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
pip install torchvision 

### cider
https://github.com/amds123/cider

http://deepai.pro/words/coco-train-idxs.p

http://deepai.pro/words/coco-train-words.p

### pytorch_fft
https://github.com/amds450076077/pytorch_fft
python setup.py install

### warp-ctc
https://github.com/amds450076077/warp-ctc

### model-zoo
https://github.com/amds450076077/tensorflow-model-zoo.torch

### install tensorboard
https://github.com/torrvision/crayon

### coco-caption
https://github.com/amds123/coco-caption.git

### AI_Challenger
https://github.com/AIChallenger/AI_Challenger.git


./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014








