import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_opt():

    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--input_json', type=str, default='/media/amds/data2/dataset/mscoco/data.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_h5', type=str, default='/media/amds/data2/dataset/mscoco/data.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_anno', type=str, default='/media/amds/data2/dataset/mscoco/anno.json',
                        help='path to the anno containing the preprocessed dataset')
    parser.add_argument('--images_root', default='/media/amds/data3/dataset/mscoco',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    parser.add_argument('--input_cnn_resnet152', type=str, default='/media/amds/data2/dataset/resnet/resnet152-b121ed2d.pth',
                        help='path to the cnn')
    parser.add_argument('--input_cnn_resnet200', type=str, default='/media/amds/data2/dataset/resnet/resnet_200_cpu.pth',
                        help = 'path to the cnn')
    parser.add_argument('--input_cnn_resnext_101_32x4d', type=str,
                        default='/media/amds/data2/dataset/resnet/resnext_101_32x4d.pth',
                        help='path to the cnn')
    parser.add_argument('--input_cnn_resnext_101_64x4d', type=str,
                        default='/media/amds/data2/dataset/resnet/resnext_101_64x4d.pth',
                        help='path to the cnn')
    parser.add_argument('--input_cnn_inceptionresnetv2', type=str,
                        default='/media/amds/data2/dataset/inception/inceptionresnetv2-d579a627.pth',
                        help='path to the cnn inceptionresnetv2')
    parser.add_argument('--input_cnn_inceptionv4', type=str,
                        default='/media/amds/data2/dataset/inception/inceptionv4-97ef9c30.pth',
                        help='path to the cnn inceptionv4')

    # bottom up attention
    parser.add_argument('--use_bu_att', type=str2bool, default=False,
                        help='use bottom up attention')
    parser.add_argument('--input_bu', type=str, default='/media/amds/data2/dataset/mscoco/anno.json',
                        help='path to bu')
    parser.add_argument('--bu_size', type=int, default=36,
                        help='path to bu size')
    parser.add_argument('--bu_feat_size', type=int, default=2048,
                        help='path to bu feat size')
    parser.add_argument('--bu_num', type=int, default=1,
                        help='bu layer of the LSTM')
    parser.add_argument('--use_image', type=str2bool, default=True,
                        help='image feature')

    # loss
    parser.add_argument('--loss_weight_start', type=float, default=1,
                        help='control the start weight of loss')
    parser.add_argument('--loss_weight_stop', type=float, default=1,
                    help='control the stop weight of loss')
    parser.add_argument('--loss_weight_type', type=int, default=0,
                        help='0 weight, 1 half')

    # use proj mul
    parser.add_argument('--use_proj_mul', type=str2bool, default=False,
                        help='')

    # data aug
    parser.add_argument('--sample_rate', type=float, default=0,
                        help='control the stop weight of loss')
    parser.add_argument('--data_norm', type=str2bool, default=True,
                        help='control the stop weight of loss')
    parser.add_argument('--use_mirror', type=str2bool, default=True,
                    help='use mirror')
    parser.add_argument('--img_padding_max', type=int, default=50,
                        help='image apdding max')
    parser.add_argument('--use_heavy_aug', type=str2bool, default=False,
                        help='use_heavy_aug')

    # tensorboard
    parser.add_argument('--use_tensorboard', type=str2bool, default=False,
                        help='use tensorboard to show the log')
    parser.add_argument('--tensorboard_type', type=int, default=0,
                        help='0 local, 1 remote')
    parser.add_argument('--tensorboard_ip', type=str, default="127.0.0.1",
                        help='the ip of the tensorboard server')
    parser.add_argument('--tensorboard_port', type=int, default=8889,
                        help='the port of the tensorboard server')
    parser.add_argument('--tensorboard_for_train_every', type=int, default=1,
                        help='')

    #
    parser.add_argument('--use_linear', type=str2bool, default=True,
                        help='use tensorboard to show the log')
    parser.add_argument('--use_linear_type', type=int, default=0,
                        help='')
    parser.add_argument('--cnn_use_linear', type=str2bool, default=False,
                        help='use tensorboard to show the log')
    parser.add_argument('--use_reviewnet', type=int, default=0,
                        help='use reviewnet')
    parser.add_argument('--review_length', type=int, default=8,
                        help='use reviewnet')


    parser.add_argument('--start_from', type=str, default='',
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--start_from_best', type=str, default='',
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'infos.pkl'         : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='print the time')

    # Model settings
    parser.add_argument('--cnn_model', type=str, default="pre_get",
                        help='resnet_152, '
                             'resnet_152_rnn, '
                             'resnet_200, '
                             'resnext_101_32x4d,'
                             'resnext_101_32x4d_7, '
                             'resnext_101_64x4d, '
                             'inceptionresnetv2, '
                             'inceptionv4')

    parser.add_argument('--caption_model', type=str, default="MoreAttenModel",
                    help='ShowTell, '
                         'ShowAttenTell, '
                         'DoubleAttenSC, '
                         'Mixer, '
                         'SCST, '
                         'BiShowAttenTellModel,'
                         'DoubleAttenMModel,'
                         'ITModel,'
                         'ShowTellPhraseModel,'
                         'MoreAttenModel,'
                         'MoreSupModel,' # current best model
                         'MoreSupPhraseModel,'
                         'ShowAttenTellPhraseModel,'
                         'ShowAttenTellPhraseBuModel,'
                         'TopDownAttenModel,'
                         'ShowAttenTellPhraseRegionModel,'
                         'ShowAttenTellPhraseRegionBuModel,'
                         'MoreSupBuModel,'
                         'ShowAttenTellPhraseBuACModel,'
                         'TransformerModel,'
                         'TransformerEDModel,'
                         'TransformerMoreSupModel,'
                         'TransformerCModel,'
                         'TransformerCMModel,'
                         'TransformerCMWGModel,'
                         'TransformerCMSModel')

    parser.add_argument('--rnn_type', type=str, default='LSTM_DOUBLE_ATT_STACK_PARALLEL_NEW',
                    help='LSTM, LSTM_SOFT_ATT, LSTM_DOUBLE_ATT, '
                         'LSTM_DOUBLE_ATT_RELU, '
                         'LSTM_SOFT_ATT_STACK, '
                         'LSTM_DOUBLE_ATT_STACK, '
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_POLICY,' 
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_BN,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_BN_RELU,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_SET,'
                         'GRU_DOUBLE_ATT_STACK_PARALLEL_DROPOUT,'
                         'LSTM_IT_ATT,'
                         'LSTM_IT_ATT_COMBINE,'
                         'CONV_LSTM,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_NEW,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_NEW,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT,'  # BEST
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_NEW,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_A,'
                         'LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT,'
                         'LSTM_SOFT_ATT_SPP,'
                         'LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_BU,'
                         'LSTM_WITH_TOP_DOWN_ATTEN,'
                         'LSTM_C_S_ATT_STACK_PARALLEL_WITH_WEIGHT_BU,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT,'
                         'LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_BU,'
                         'LSTM_SOFT_ATT_STACK_PARALLEL_WITH_FC_WEIGHT')

    parser.add_argument('--value_type', type=str, default='LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_BU_VALUE',
                        help='LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_BU_VALUE')

    # lmdb path
    parser.add_argument('--use_pre_feat', type=str2bool, default=False,
                        help='')
    parser.add_argument('--path_lmdb', type=str, default='',
                        help='')
    parser.add_argument('--use_lmdb_img', type=str2bool, default=False,
                        help='')
    parser.add_argument('--path_lmdb_img', type=str, default='',
                        help='')

    # memory image
    parser.add_argument('--use_memory_img', type=str2bool, default=False,
                        help='')

    # memcache img
    parser.add_argument('--use_memcache_img', type=str2bool, default=False,
                        help='')
    parser.add_argument('--memcache_host', type=str, default="127.0.0.1:11211",
                        help='')



    # trick layers
    parser.add_argument('--use_gated_layer', type=int, default=0,
                        help='tricks for gated layer'
                             '0, None'
                             '1, GatedTanh')
    parser.add_argument('--use_linear_embedding', type=int, default=0,
                        help='tricks for embedding layer'
                             '0, None'
                             '1, embedding')
    parser.add_argument('--cider_idxs', type=str, default='coco-train-idxs',
                        help='')
    parser.add_argument('--eval_metric', type=str, default='CIDEr',
                        help='CIDEr, ROUGE_L, Bleu_1, Bleu_2, Bleu_3, Bleu_4')
    parser.add_argument('--relu_type', type=int, default=0,
                        help='0 original, 1 normal')

    # embedding type
    parser.add_argument('--word_embedding_type', type=int, default=0,
                        help='0 embedding, 1 embedding_with_bais')

    # feat size setting
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg, 1536 for inception')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg, 1536 for inception')
    parser.add_argument('--att_size', type=int, default=49,
                        help='49 for resnet152, 196 for mul, 64 for inception')
    parser.add_argument('--spp_num', type=int, default=6,
                        help='')
    parser.add_argument('--image_size', type=int, default=224,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--pool_size', type=int, default=7,
                    help='size of the pool, 7 for resnet, 8 for inception')
    parser.add_argument('--att_pool_size', type=int, default=1,
                    help='size of the pool, 1 for default')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--num_parallels', type=int, default=1,
                        help='parallels of layers in the RNN')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--is_every_eval', type=str2bool, default=True,
                    help='')

    # libinear
    parser.add_argument('--use_bilinear', type=str2bool, default=False,
                        help='use bilinear')
    parser.add_argument('--bilinear_output', type=int, default=1000,
                        help='the output of bilinear')

    # memory
    parser.add_argument('--memory_num_hop', type=int, default=1,
                    help='')

    # rnn_size_list
    parser.add_argument('--rnn_size_list', type=list, default=[512, 768, 1024],
                    help='size of the rnn in number of hidden nodes in each layer')


    parser.add_argument('--block_num', type=int, default=1,
                        help='block_num LSTM Linear blcok num')
    parser.add_argument('--context_len', type=int, default=4,
                        help='context_len for phrase embedding num')
    parser.add_argument('--rnn_atten', type=str, default='NONE',
                        help='NONE, ATT_LSTM')

    # gram and prob weight
    parser.add_argument('--gram_num', type=int, default=0,
                        help='gram_num for word word num')
    parser.add_argument('--word_gram_num', type=int, default=4,
                        help='gram_num for word word num')
    parser.add_argument('--phrase_gram_num', type=int, default=4,
                        help='gram_num for word phrase num')
    parser.add_argument('--conv_gram_num', type=int, default=4,
                        help='gram_num for word conv num')
    parser.add_argument('--phrase_type', type=int, default=1,
                        help='0 none, 1 phrase, 2 conv, 3 phrase-conv')
    parser.add_argument('--mil_type', type=int, default=2,
                        help='0 max, 1 nor, 2 mean')
    parser.add_argument('--prob_weight_alpha', type=float, default=0.9,
                        help='prob weight alpha')
    # prob
    parser.add_argument('--logprob_pool_type', type=int, default=0,
                        help='logprob_pool_type,0 mean, 1 max, 2 top')

    parser.add_argument('--use_prob_weight', type=str2bool, default=False,
                        help='is use prob weight')

    # input layers
    parser.add_argument('--word_input_layer', type=int, default=2,
                        help='')
    parser.add_argument('--att_input_layer', type=int, default=1,
                        help='')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--finetune_cnn_after', type=int, default=-1,
                       help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--finetune_cnn_type', type=int, default=0,
                        help='0 all, 1 part')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')


    # ctc_start
    parser.add_argument('--ctc_start', type=int, default=-1,
                        help='at what iteration to start ctc learning ? (-1 = dont) (in epoch)')

    #Optimization: for the Reinforce Learning
    parser.add_argument('--reinforce_start', type=int, default=-1,
                        help='at what iteration to start reinforce learning ? (-1 = dont) (in epoch)')
    parser.add_argument('--reinforce_type', type=int, default=1,
                        help='0 policy gradient, 1 self-critical, 2 actor-critic')
    parser.add_argument('--mix_start', type=int, default=-1,
                        help='at what iteration to start reinforce learning ? (-1 = dont) (in epoch)')
    parser.add_argument('--path_cider', type=str, default='/home/scw4750/caption/cider',
                        help='')
    parser.add_argument('--path_idxs', type=str, default='/home/scw4750/caption/dataset/mscoco',
                        help='')

    parser.add_argument('--rl_critic_start', type=int, default=-1,
                        help='')
    parser.add_argument('--rl_actor_critic_start', type=int, default=-1,
                        help='')

    parser.add_argument('--rl_alpha_start', type=float, default=0.8,
                        help='')
    parser.add_argument('--rl_alpha_recent_start', type=float, default=0.8,
                        help='')
    parser.add_argument('--rl_alpha_recent_num', type=int, default=7500,
                        help='')
    parser.add_argument('--rl_alpha_type', type=int, default=1,
                        help='0 start, 1 recent start')
    parser.add_argument('--rl_beta', type=float, default=1.0,
                        help='')
    parser.add_argument('--rl_gamma', type=float, default=1.0,
                        help='')
    parser.add_argument('--rl_use_gamma', type=str2bool, default=False,
                        help='')
    parser.add_argument('--is_eval_start', type=str2bool, default=False,
                        help='')
    parser.add_argument('--rl_metric', type=str, default='CIDEr',
                        help='CIDEr, ROUGE_L, Bleu_4, AVG')

    parser.add_argument('--rl_beta_incre_start', type=float, default=-1,
                        help='beta steply incresing with iterations, -1 for close')
    parser.add_argument('--rl_beta_incre_iters_every', type=float, default=2000,
                        help='beta steply incresing every n iterations')
    parser.add_argument('--rl_beta_incre_every_add', type=float, default=0.01,
                        help='beta steply incresing, each time add x')
    parser.add_argument('--is_beta_incre_linear', type=str2bool, default=False,
                        help='increase beta linearly')
    parser.add_argument('--rl_beta_incre_max', type=float, default=0.95,
                        help='max value of beta')

    parser.add_argument('--rl_hard_type', type=int, default=0,
                        help='0 none, 1 daptive')
    parser.add_argument('--rl_hard_alpha', type=float, default=1,
                        help='0 none, 1 daptive')
    parser.add_argument('--rl_hard_reward', type=float, default=3.78,
                        help='')

    parser.add_argument('--rl_mask_type', type=int, default=0,
                        help='the mask type for SCST')

    #Optimization: for the Language Model
    parser.add_argument('--use_auto_learning_rate', type=int, default=0,
                        help='0 None,'
                             '1 use')
    parser.add_argument('--auto_start_from_best', type=str, default='',
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                                'infos.pkl'         : configuration;
                                'checkpoint'        : paths to model file(s) (created by tf).
                                                      Note: this file contains absolute paths, be careful when moving files around;
                                'model.ckpt-*'      : file(s) with model definition (created by tf)
                            """)
    parser.add_argument('--auto_early_stop_cnt', type=int, default=3,
                        help='')
    parser.add_argument('--auto_early_stop_score', type=float, default=0.5,
                        help='')


    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.8,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')

    #Optimization: for the CNN
    parser.add_argument('--cnn_optim', type=str, default='adam',
                    help='optimization to use for CNN')
    parser.add_argument('--cnn_optim_alpha', type=float, default=0.8,
                    help='alpha for momentum of CNN')
    parser.add_argument('--cnn_optim_beta', type=float, default=0.999,
                    help='beta for momentum of CNN')
    parser.add_argument('--cnn_learning_rate', type=float, default=5e-5,
                    help='learning rate for the CNN')
    parser.add_argument('--cnn_weight_decay', type=float, default=0,
                    help='L2 weight decay just for the CNN')

    parser.add_argument('--cnn_learning_rate_decay_start', type=int, default=-1,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--cnn_learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--cnn_learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')

    # Evaluation/Checkpointing
    parser.add_argument('--is_compute_val_loss', type=int, default=0,
                        help='0 not, 1 compute')
    parser.add_argument('--val_split', type=str, default='test',
                    help='use to valid result')
    parser.add_argument('--val_images_use', type=int, default=2,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=10,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--save_snapshot_every', type=int, default=-1,
                        help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='../caption_checkpoint/resnet152/',
                    help='directory to store checkpointed models')
    parser.add_argument('--checkpoint_best_path', type=str, default='../caption_checkpoint_best/resnet152/',
                        help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')
    parser.add_argument('--is_load_infos', type=int, default=1,
                        help='')
    parser.add_argument('--coco_caption_path', type=str, default='/media/amds/data1/code/coco-caption',
                    help='coco_caption_path')
    parser.add_argument('--eval_result_path', type=str, default='../caption_result/resnet152/',
                    help='eval_result_path')

    # aic data
    parser.add_argument('--is_aic_data', type=str2bool, default=False,
                        help='is_aic_data')
    parser.add_argument('--aic_caption_path', type=str, default='/media/amds/data3/caption/AI_Challenger/Evaluation/caption_eval/coco_caption',
                        help='aic_caption_path')

    # reward
    parser.add_argument('--cureward_gamma', type=float, default=0.99,
                    help='')
    parser.add_argument('--reward_gamma', type=float, default=2,
                    help='')
    parser.add_argument('--reward_base', type=float, default=0.5,
                    help='')

    # refine_num
    parser.add_argument('--refine_num', type=int, default=1,
                        help='')

    # misc
    parser.add_argument('--id', type=str, default='dasc_rf_1',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')

    # transformer
    parser.add_argument('--model_size', type=int, default=512,
                        help='model size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                            decoded sentences""")
    parser.add_argument('--adaptive_size', type=int, default=4,
                        help='adaptive_size')
    parser.add_argument('--head_size', type=int, default=8,
                        help='')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='')
    parser.add_argument('--k_size', type=int, default=64,
                        help='')
    parser.add_argument('--v_size', type=int, default=64,
                        help='')
    parser.add_argument('--inner_layer_size', type=int, default=2048,
                        help='')
    parser.add_argument('--is_show_result', type=str2bool, default=False,
                        help='')
    parser.add_argument('--transformer_decoder_type', type=str, default='ImageDecoder',
                        help='ImageDecoder,'
                             'ImageAttentionDecoder,'
                             'ImageMoreSupDecoder,'
                             'ImageMoreSupAttentionDecoder,'
                             'ImageCDecoder,'
                             'ImageCMDecoder,'
                             'ImageCMWGDecoder,'
                             'ImageCMSDecoder')
    parser.add_argument('--transformer_decoder_layer_type', type=str, default='DecoderLayer',
                        help='DecoderLayer,'
                             'AttentionDecoderLayer,'
                             'CDecoderLayer')
    parser.add_argument('--image_embed_type', type=int, default=1,
                        help='image_embed_type')
    parser.add_argument('--n_layers_output', type=int, default=1,
                        help='n_layers_output')
    parser.add_argument('--norm_type', type=int, default=0,
                        help='0, norm'
                             '1, batch_norm')
    parser.add_argument('--cnn_bn_drop_type', type=int, default=0,
                        help='0, None'
                             '1, drop,'
                             '2, bn_drop')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args