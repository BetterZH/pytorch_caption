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

# train transformer
### Captioning Transformer with Stacked Attention Modules
Paper Link [https://www.mdpi.com/2076-3417/8/5/739/html]

the script
```
device_id=$1
tensorboard_ip=`cat tensorboard_ip`
tensorboard_port=`cat tensorboard_port`
cd ../pytorch_caption
CUDA_VISIBLE_DEVICES=$device_id python train.py --input_json ../dataset/caption_datasets/data_coco.json \
--input_h5 ../dataset/caption_datasets/data_coco.h5 \
--input_anno ../dataset/caption_datasets/anno_coco.json \
--input_cnn_resnet152 ../dataset/resnet/resnet152-b121ed2d.pth \
--input_cnn_resnet200 ../dataset/resnet/resnet_200_cpu.pth \
--input_cnn_resnext_101_32x4d ../dataset/resnet/resnext_101_32x4d.pth \
--input_cnn_resnext_101_64x4d ../dataset/resnet/resnext_101_64x4d.pth \
--input_cnn_inceptionresnetv2 ../dataset/inception/inceptionresnetv2-d579a627.pth \
--input_cnn_inceptionv4 ../dataset/inception/inceptionv4-97ef9c30.pth \
--images_root ../images \
--coco_caption_path ../dataset/coco-caption \
--is_aic_data False \
--eval_metric CIDEr \
--aic_caption_path /home/amds/caption/AI_Challenger/Evaluation/caption_eval/coco_caption \
--path_cider ../dataset/cider \
--path_idxs ../dataset/words \
--cider_idxs coco-train-idxs \
--val_images_use 5000 \
--save_checkpoint_every 2500 \
--save_snapshot_every 500 \
--batch_size 16 \
--caption_model TransformerCModel \
--transformer_decoder_type ImageCDecoder \
--transformer_decoder_layer_type CDecoderLayer \
--cnn_model resnext_101_64x4d \
--checkpoint_path ../caption_checkpoint/transformer \
--checkpoint_best_path ../caption_checkpoint_best/transformer \
--finetune_cnn_after -1 \
--finetune_cnn_type 0 \
--val_split 'test' \
--start_from '' \
--start_from_best '../caption_checkpoint_best/transformer' \
--eval_result_path ../caption_result/transformer \
--beam_size 2 \
--learning_rate 1e-5 \
--cnn_learning_rate 1e-6 \
--use_tensorboard False \
--tensorboard_ip $tensorboard_ip \
--tensorboard_port $tensorboard_port \
--reinforce_start -1 \
--reinforce_type 1 \
--model_size 512 \
--fc_feat_size 2048 \
--att_feat_size 2048 \
--att_size 49 \
--input_encoding_size 512 \
--rnn_size 512 \
--image_size 224 \
--pool_size 7 \
--is_eval_start False \
--drop_prob_lm 0.7 \
--load_best_score 1 \
--adaptive_size 4 \
--head_size 8 \
--n_layers 6 \
--k_size 64 \
--v_size 64 \
--n_layers_output 6 \
--inner_layer_size 2048 \
--is_show_result False \
--logprob_pool_type 0 \
--verbose 0 \
--use_heavy_aug True \
--id tfc_64a4n6
```
