import os

import torch

import cnn.cnn_utils as cnn_utils
from cnn import resnet
from cnn import resnet200
from cnn import resnext_101_32x4d
from cnn import resnext_101_32x4d_7
from cnn import resnext_101_64x4d
from cnn import inceptionresnetv2
from cnn import inceptionv4


# resnext_101_64x4d.t7
# https://s3.amazonaws.com/resnext/imagenet_models/resnext_101_64x4d.t7
# resnext_101_32x4d.t7
# https://s3.amazonaws.com/resnext/imagenet_models/resnext_101_32x4d.t7
# resnet-200.t7
# https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7
# inceptionv4
# http://webia.lip6.fr/~cadene/Downloads/inceptionv4-97ef9c30.pth
# inceptionresnetv2
# http://webia.lip6.fr/~cadene/Downloads/inceptionresnetv2-d579a627.pth


def is_transformer(caption_model):
    if caption_model in ['TransformerModel',
                         'TransformerEDModel',
                         'TransformerNewModel',
                         'TransformerMoreSupModel']:
        return True
    return False

def is_ctransformer(caption_model):
    if caption_model in ['TransformerCModel',
                         'TransformerCMModel',
                         'TransformerCMWGModel',
                         'TransformerCMSModel']:
        return True
    return False

def has_bu(caption_model):
    if caption_model in ['ShowAttenTellPhraseBuModel',
                         'ShowAttenTellPhraseBuRefineModel',
                         'MoreSupBuModel',
                         'ShowAttenTellPhraseBuACModel']:
        return True
    return False

def is_only_bu_feat(caption_model):
    if caption_model in ['TopDownAttenModel']:
        return True
    return False

def is_only_fc_feat(caption_model):
    if caption_model in ['ShowTell', 'ShowTellPhraseModel']:
        return True
    return False

def has_sub_regions(caption_model):
    if caption_model in ['ShowAttenTellPhraseRegionModel']:
        return True
    return False

def has_sub_region_bu(caption_model):
    if caption_model in ['ShowAttenTellPhraseRegionBuModel']:
        return True
    return False

def is_only_att_feat(caption_model):
    if caption_model in ['MoreAttenModel',
                         'TransformerModel',
                         'TransformerModel',
                         'TransformerEDModel',
                         'TransformerNewModel',
                         'TransformerMoreSupModel']:
        return True
    return False

def is_mul_out_with_weight(caption_model):
    if caption_model in ['MoreSupWeightModel']:
        return True
    return False

def is_mul_out(caption_model):
    if caption_model in ['MoreSupModel',
                         'MoreSupBuModel',
                         'TransformerMoreSupModel',
                         'TransformerCMSModel']:
        return True
    return False

def is_prob_weight(caption_model):
    if caption_model in ['ShowAttenTellPhraseModel',
                         'MoreSupWeightModel',
                         'ShowAttenTellPhraseBuModel',
                         'ShowAttenTellPhraseRegionModel',
                         'ShowAttenTellPhraseRegionBuModel',
                         'ShowAttenTellPhraseBuACModel',
                         'TransformerCMWGModel']:
        return True
    return False

def is_prob_weight_mul_out(caption_model):
    if caption_model in ['ShowAttenTellPhraseBuRefineModel']:
        return True
    return False

def is_inception(cnn_model):
    if cnn_model in ['inceptionresnetv2', 'inceptionv4']:
        return True
    return False

def is_pre_get(cnn_model):
    if cnn_model in ['pre_get']:
        return True
    return False

def clean_key(key):
    if key.startswith('module.'):
        key = key.partition('module.')[2]
    return key

def setup(opt):

    print("caption model: " + opt.caption_model)

    if opt.caption_model == 'ShowTell':
        from misc.ShowTellModel import ShowTellModel
        model = ShowTellModel(opt)
    elif opt.caption_model == 'ShowAttenTell':
        from misc.ShowAttenTellModel import ShowAttenTellModel
        model = ShowAttenTellModel(opt)
    elif opt.caption_model == 'DoubleAttenSC':
        from misc.DoubleAttenSCModel import DoubleAttenSCModel
        model = DoubleAttenSCModel(opt)
    elif opt.caption_model == 'Mixer':
        from misc.MixerModel import MixerModel
        model = MixerModel(opt)
    elif opt.caption_model == 'SCST':
        from misc.SCSTModel import SCSTModel
        model = SCSTModel(opt)
    elif opt.caption_model == 'BiShowAttenTellModel':
        from misc.BiShowAttenTellModel import BiShowAttenTellModel
        model = BiShowAttenTellModel(opt)
    elif opt.caption_model == 'DoubleAttenMModel':
        from misc.DoubleAttenMModel import DoubleAttenMModel
        model = DoubleAttenMModel(opt)
    elif opt.caption_model == 'ITModel':
        from misc.ITModel import ITModel
        model = ITModel(opt)
    elif opt.caption_model == 'ShowTellPhraseModel':
        from misc.ShowTellPhraseModel import ShowTellPhraseModel
        model = ShowTellPhraseModel(opt)
    elif opt.caption_model == 'MoreAttenModel':
        from misc.MoreAttenModel import MoreAttenModel
        model = MoreAttenModel(opt)
    elif opt.caption_model == 'MoreSupModel':
        from misc.MoreSupModel import MoreSupModel
        model = MoreSupModel(opt)
    elif opt.caption_model == 'MoreSupPhraseModel':
        from misc.MoreSupPhraseModel import MoreSupPhraseModel
        model = MoreSupPhraseModel(opt)
    elif opt.caption_model == 'ShowAttenTellPhraseModel' or opt.caption_model == 'ShowAttenTellPhraseRegionModel':
        from misc.ShowAttenTellPhraseModel import ShowAttenTellPhraseModel
        model = ShowAttenTellPhraseModel(opt)
    elif opt.caption_model == 'MoreSupWeightModel':
        from misc.MoreSupWeightModel import MoreSupWeightModel
        model = MoreSupWeightModel(opt)
    elif opt.caption_model == 'ShowAttenTellPhraseBuModel' or opt.caption_model == 'ShowAttenTellPhraseRegionBuModel':
        from misc.ShowAttenTellPhraseBuModel import ShowAttenTellPhraseBuModel
        model = ShowAttenTellPhraseBuModel(opt)
    elif opt.caption_model == 'TopDownAttenModel':
        from misc.TopDownAttenModel import TopDownAttenModel
        model = TopDownAttenModel(opt)
    elif opt.caption_model == 'ShowAttenTellPhraseBuRefineModel':
        from misc.ShowAttenTellPhraseBuRefineModel import ShowAttenTellPhraseBuRefineModel
        model = ShowAttenTellPhraseBuRefineModel(opt)
    elif opt.caption_model == 'MoreSupBuModel':
        from misc.MoreSupBuModel import MoreSupBuModel
        model = MoreSupBuModel(opt)
    elif opt.caption_model == 'ShowAttenTellPhraseBuACModel':
        from misc.ShowAttenTellPhraseBuACModel import ShowAttenTellPhraseBuACModel
        model = ShowAttenTellPhraseBuACModel(opt)
    elif opt.caption_model == 'TripleAttenModel':
        from misc.TripleAttenModel import TripleAttenModel
        model = TripleAttenModel(opt)
    elif opt.caption_model == 'TransformerModel':
        from misc.TransformerModel import TransformerModel
        model = TransformerModel(opt)
    elif opt.caption_model == 'TransformerEDModel':
        from misc.TransformerEDModel import TransformerEDModel
        model = TransformerEDModel(opt)
    elif opt.caption_model == 'TransformerMoreSupModel':
        from misc.TransformerMoreSupModel import TransformerMoreSupModel
        model = TransformerMoreSupModel(opt)
    elif opt.caption_model == 'TransformerCModel':
        from misc.TransformerCModel import TransformerCModel
        model = TransformerCModel(opt)
    elif opt.caption_model == 'TransformerCMModel':
        from misc.TransformerCMModel import TransformerCMModel
        model = TransformerCMModel(opt)
    elif opt.caption_model == 'TransformerCMWGModel':
        from misc.TransformerCMWGModel import TransformerCMWGModel
        model = TransformerCMWGModel(opt)
    elif opt.caption_model == 'TransformerCMSModel':
        from misc.TransformerCMSModel import TransformerCMSModel
        model = TransformerCMSModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    start_from_best = vars(opt).get('start_from_best', None).strip()
    start_from = vars(opt).get('start_from', None).strip()

    if start_from_best is not None and len(start_from_best) > 0 and opt.id.endswith('.pkl'):
        # ensemble, id is infos okl file name
        model_name = opt.id.replace('_infos', '_model').replace('.pkl', '.pth')
        path_model = os.path.join(start_from_best, model_name)
        print(path_model)
        pretrained_dict = torch.load(path_model)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        print('model rely_type', model.relu_type)
        for k, v in pretrained_dict.items():
            if not model_dict.has_key(k) or model_dict[k].size() != pretrained_dict[k].size():
                print('model_dict.has_key ' + str(k), model_dict.has_key(k))
                print('pretrained_dict', k)
                del pretrained_dict[k]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    elif start_from_best is not None and len(start_from_best) > 0:
        path_model = os.path.join(start_from_best, opt.id + '_model_best.pth')
        print(path_model)

        pretrained_dict = torch.load(path_model)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()

        # proj_weight = torch.load('/home/amds/att_weight_gram/soft_att_weight_1_model_best.pth')
        # print(proj_weight.keys())
        # pretrained_dict['core.proj_weight.weight'] = proj_weight['core.proj_weight.weight']
        # pretrained_dict['core.proj_weight.bias'] = proj_weight['core.proj_weight.bias']

        # for k, v in pretrained_dict.items():
        #     print(k, v.size(), model_dict[k].size(), v.size(), model_dict[k].size()==v.size())

        for k, v in pretrained_dict.items():
            if not model_dict.has_key(k) or model_dict[k].size() != pretrained_dict[k].size():
                print(k)
                del pretrained_dict[k]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif start_from is not None and len(start_from) > 0:
        path_model = os.path.join(start_from, opt.id + '_model.pth')
        print(path_model)
        pretrained_dict = torch.load(path_model)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def load_cnn_model(model, opt):
    if is_only_fc_feat(opt.caption_model):
        if opt.cnn_use_linear:
            model_cnn = cnn_utils.resnet_fc_linear(model, opt)
        else:
            model_cnn = cnn_utils.resnet_fc(model, opt)
    elif is_only_att_feat(opt.caption_model):
        model_cnn = cnn_utils.resnet_att(model, opt)
    elif has_sub_regions(opt.caption_model):
        model_cnn = cnn_utils.resnet_with_conv_regions(model, opt)
    elif has_sub_region_bu(opt.caption_model):
        model_cnn = cnn_utils.resnet_with_att_conv_regions(model, opt)
    else:
        if opt.cnn_use_linear:
            model_cnn = cnn_utils.resnet_fc_att_linear(model, opt)
        else:
            model_cnn = cnn_utils.resnet_fc_att(model, opt)
    return model_cnn

def load_cnn_model_identity(opt):
    model = None
    if is_only_fc_feat(opt.caption_model):
        if opt.cnn_use_linear:
            model_cnn = cnn_utils.resnet_fc_linear(model, opt)
        else:
            model_cnn = cnn_utils.resnet_fc(model, opt)
    elif is_only_att_feat(opt.caption_model):
        model_cnn = cnn_utils.resnet_att(model, opt)
    else:
        if opt.cnn_use_linear:
            model_cnn = cnn_utils.resnet_fc_att_linear(model, opt)
        else:
            model_cnn = cnn_utils.resnet_fc_att(model, opt)
    return model_cnn


def setup_default_cnn(opt):

    if opt.cnn_model == "resnet_152":
        model = resnet.resnet152()
        input_cnn = opt.input_cnn_resnet152
    elif opt.cnn_model == "resnet_200":
        model = resnet200.resnet200
        input_cnn = opt.input_cnn_resnet200
    elif opt.cnn_model == "resnext_101_32x4d":
        model = resnext_101_32x4d.resnext_101_32x4d
        input_cnn = opt.input_cnn_resnext_101_32x4d
    elif opt.cnn_model == "resnext_101_32x4d_7":
        model = resnext_101_32x4d_7.resnext_101_32x4d
        input_cnn = opt.input_cnn_resnext_101_32x4d
    elif opt.cnn_model == "resnext_101_64x4d":
        model = resnext_101_64x4d.resnext_101_64x4d
        input_cnn = opt.input_cnn_resnext_101_64x4d
    elif opt.cnn_model == "inceptionresnetv2":
        model = inceptionresnetv2.inceptionresnetv2()
        input_cnn = opt.input_cnn_inceptionresnetv2
    elif opt.cnn_model == "inceptionv4":
        model = inceptionv4.inceptionv4()
        input_cnn = opt.input_cnn_inceptionv4
    else:
        raise Exception("cnn model not supported: {}".format(opt.cnn_model))

    print(input_cnn)

    pretrained_dict = torch.load(input_cnn)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model




def setup_cnn(opt):

    if is_pre_get(opt.cnn_model):
        print('pre get')
        model_cnn = load_cnn_model_identity(opt)
        return model_cnn
    elif opt.cnn_model == "resnet_152":
        model = resnet.resnet152()
        input_cnn = opt.input_cnn_resnet152
    elif opt.cnn_model == "resnet_200":
        model = resnet200.resnet200
        input_cnn = opt.input_cnn_resnet200
    elif opt.cnn_model == "resnext_101_32x4d":
        model = resnext_101_32x4d.resnext_101_32x4d
        input_cnn = opt.input_cnn_resnext_101_32x4d
    elif opt.cnn_model == "resnext_101_32x4d_7":
        model = resnext_101_32x4d_7.resnext_101_32x4d
        input_cnn = opt.input_cnn_resnext_101_32x4d
    elif opt.cnn_model == "resnext_101_64x4d":
        model = resnext_101_64x4d.resnext_101_64x4d
        input_cnn = opt.input_cnn_resnext_101_64x4d
    elif opt.cnn_model == "inceptionresnetv2":
        model = inceptionresnetv2.inceptionresnetv2()
        input_cnn = opt.input_cnn_inceptionresnetv2
    elif opt.cnn_model == "inceptionv4":
        model = inceptionv4.inceptionv4()
        input_cnn = opt.input_cnn_inceptionv4
    else:
        raise Exception("cnn model not supported: {}".format(opt.cnn_model))


    start_from_best = vars(opt).get('start_from_best', None).strip()
    start_from = vars(opt).get('start_from', None).strip()

    if start_from_best is not None and len(start_from_best) > 0 and opt.id.endswith('.pkl'):
        # ensemble, id is infos okl file name
        model_cnn_name = opt.id.replace('_infos', '_model_cnn').replace('.pkl', '.pth')
        model_cnn = load_cnn_model(model, opt)
        model_path = os.path.join(start_from_best, model_cnn_name)
        print(model_path)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
        model_cnn.load_state_dict(pretrained_dict)

    elif start_from_best is not None and len(start_from_best) > 0:
        model_cnn = load_cnn_model(model, opt)
        model_path = os.path.join(start_from_best, opt.id + '_model_cnn_best.pth')
        print(model_path)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
        model_cnn.load_state_dict(pretrained_dict)
    elif start_from is not None and len(start_from) > 0:
        model_cnn = load_cnn_model(model, opt)
        model_path = os.path.join(opt.start_from, opt.id + '_model_cnn.pth')
        print(model_path)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}
        model_cnn.load_state_dict(pretrained_dict)
    else:

        print("cnn: " + input_cnn)

        pretrained_dict = torch.load(input_cnn)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        model_cnn = load_cnn_model(model, opt)

    return model_cnn


def get_fintune_layers(model_cnn, opt):

    if opt.cnn_model == "resnet_152":
        layers = [model_cnn.resnet.layer4]
    elif opt.cnn_model == "resnet_200":
        layers = [model_cnn.resnet.layer4]
    elif opt.cnn_model == "resnext_101_32x4d":
        layers = [model_cnn.resnet[7], model_cnn.resnet[6]]
    elif opt.cnn_model == "resnext_101_64x4d":
        layers = [model_cnn.resnet[7], model_cnn.resnet[6], model_cnn.resnet[5]]

    return layers

