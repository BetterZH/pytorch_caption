#pytorch_caption
1. change resnet200
2. change loss
3. add attention
4. add double attention
5. add text condition
6. add MAT
7. add reinforce learning


#idea
1. rnn as attention
2. reinforce for caption
3. GAN for caption
4. how to use loss for caption
5. how to improve caption
6. how to use data

# install pytorch
conda install pytorch torchvision -c soumith

# install tensorboard
https://github.com/torrvision/crayon

#
最近尝试的改进：
１．修改LSTM，现在LSTM单元一般是单层的，现在尝试改进为多层，并增加并联（效果提升不明显，还需要改进）
２．现在实验发现网络有过拟合现象，在开始阶段各项指标提升非常快，但是慢慢开始下降，尝试了增加relu,batch_norm,dropout,weight_decay正则项优化网络的训练
３．实验不同CNN，其中包括resnet152,resnet200,resnext-101-32*4d,resnext-101-64*4d对最终结果的影响（正在运行）
准备尝试的改进（争取在两周内验证完）：
１．尝试应用Deep Q-learning增强网络网络优化训练过程
２．尝试生成对抗网络的应用
Semantic Segmentation using Adversarial Networks是Facebook的一篇用GAN的思想去做语义分割的论文，可以借鉴相应思想构建关于Image Caption的对抗网络，
主要思想是增加一个判断网络，判断语句是生成的还是标注数据的．
３．优化词嵌入模型，现在的词嵌入相对比较简单，没有任何语义信息，准备尝试google提出的word2vec算法，改进单词嵌入阶段的词向量
４．尝试Tree-LSTM模型，改进现有的LSTM模型
参考论文Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
５．尝试用CNN网络做词向量模型
参考论文An Empirical Study of Language CNN for Image Captioning,但是该篇效果不算太好
6. 利用多尺度的思想组合LSTM