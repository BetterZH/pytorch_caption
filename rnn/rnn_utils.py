import rnn.GRU as GRU
import rnn.LSTM as LSTM
import rnn.LSTM1 as LSTM1
import rnn.LSTM2 as LSTM2
import rnn.LSTM3 as LSTM3
import rnn.LSTM4 as LSTM4
import rnn.LSTM5 as LSTM5
import rnn.LSTM6 as LSTM6

def get_value(opt):

    print('value_type ', opt.value_type)

    if opt.value_type == "LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_BU_VALUE":
        core = LSTM5.LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_BU_VALUE(opt.input_encoding_size, opt.vocab_size + 1,
                                                                     opt.num_layers,
                                                                     opt.num_parallels, opt.rnn_size, opt.att_size,
                                                                     opt.bu_size,
                                                                     dropout=opt.drop_prob_lm)
    else:
        raise Exception("value type not supported: {}".format(opt.rnn_type))

    return core


def get_lstm(opt):

    print('rnn_type ', opt.rnn_type)

    # LSTM
    if opt.rnn_type == "LSTM":
        core = LSTM.LSTM(opt.input_encoding_size, opt.vocab_size + 1,
                         opt.rnn_size, opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT":
        core = LSTM.LSTM_SOFT_ATT(opt.input_encoding_size, opt.vocab_size + 1,
                                  opt.rnn_size, opt.att_size, opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT":
        core = LSTM.LSTM_DOUBLE_ATT(opt.input_encoding_size, opt.vocab_size + 1,
                                    opt.rnn_size, opt.att_size, opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK":
        core = LSTM.LSTM_SOFT_ATT_STACK(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                        opt.rnn_size, opt.att_size, dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK":
        core = LSTM.LSTM_DOUBLE_ATT_STACK(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                          opt.rnn_size, opt.att_size, dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL":
        core = LSTM.LSTM_DOUBLE_ATT_STACK_PARALLEL(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_POLICY":
        core = LSTM.LSTM_DOUBLE_ATT_STACK_PARALLEL_POLICY(opt.input_encoding_size, opt.vocab_size + 1,
                                                          opt.num_layers,
                                                          opt.num_parallels, opt.rnn_size, opt.att_size,
                                                          dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_BN":
        core = LSTM.LSTM_DOUBLE_ATT_STACK_PARALLEL_BN(opt.input_encoding_size, opt.vocab_size + 1,
                                                      opt.num_layers,
                                                      opt.num_parallels, opt.rnn_size, opt.att_size,
                                                      dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_BN_RELU":
        core = LSTM.LSTM_DOUBLE_ATT_STACK_PARALLEL_BN_RELU(opt.input_encoding_size, opt.vocab_size + 1,
                                                           opt.num_layers,
                                                           opt.num_parallels, opt.rnn_size, opt.att_size,
                                                           dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT":
        core = LSTM.LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT(opt.input_encoding_size, opt.vocab_size + 1,
                                                           opt.num_layers,
                                                           opt.num_parallels, opt.rnn_size, opt.att_size,
                                                           dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_SET":
        core = LSTM.LSTM_DOUBLE_ATT_STACK_PARALLEL_DROPOUT_SET(opt.input_encoding_size, opt.vocab_size + 1,
                                                               opt.num_layers,
                                                               opt.num_parallels,
                                                               opt.rnn_size,
                                                               opt.rnn_size_list, opt.att_size,
                                                               dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "GRU_DOUBLE_ATT_STACK_PARALLEL_DROPOUT":
        core = GRU.GRU_DOUBLE_ATT_STACK_PARALLEL_DROPOUT(opt.input_encoding_size, opt.vocab_size + 1,
                                                         opt.num_layers,
                                                         opt.num_parallels, opt.rnn_size, opt.att_size,
                                                         dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_IT_ATT":
        core = LSTM1.LSTM_IT_ATT(opt.input_encoding_size,
                                 opt.vocab_size + 1,
                                 opt.rnn_size,
                                 opt.att_size,
                                 opt.drop_prob_lm,
                                 opt.num_layers,
                                 opt.word_input_layer,
                                 opt.att_input_layer)
    elif opt.rnn_type == "LSTM_IT_ATT_COMBINE":
        core = LSTM1.LSTM_IT_ATT_COMBINE(opt.input_encoding_size,
                                         opt.vocab_size + 1,
                                         opt.rnn_size,
                                         opt.att_size,
                                         opt.drop_prob_lm,
                                         opt.num_layers,
                                         opt.word_input_layer,
                                         opt.att_input_layer)
    elif opt.rnn_type == "FO_IT_ATT_COMBINE":
        core = LSTM1.FO_IT_ATT_COMBINE(opt.input_encoding_size,
                                       opt.vocab_size + 1,
                                       opt.rnn_size,
                                       opt.att_size,
                                       opt.drop_prob_lm,
                                       opt.num_layers,
                                       opt.word_input_layer,
                                       opt.att_input_layer)
    elif opt.rnn_type == "CONV_IT_ATT_COMBINE":
        core = LSTM1.CONV_IT_ATT_COMBINE(opt.input_encoding_size,
                                         opt.vocab_size + 1,
                                         opt.rnn_size,
                                         opt.att_size,
                                         opt.drop_prob_lm,
                                         opt.num_layers,
                                         opt.word_input_layer,
                                         opt.att_input_layer)
    elif opt.rnn_type == "CONV_LSTM":
        core = LSTM1.CONV_LSTM(opt.input_encoding_size, opt.vocab_size + 1,
                               opt.rnn_size, opt.drop_prob_lm, opt.num_layers, opt.block_num, opt.use_proj_mul)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_NEW":
        core = LSTM1.LSTM_DOUBLE_ATT_STACK_PARALLEL(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT":
        core = LSTM1.LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_NEW":
        core = LSTM1.LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_NEW(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT":
        core = LSTM2.LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_BU":
        core = LSTM2.LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_BU(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size, opt.bu_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_NEW":
        core = LSTM2.LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_NEW(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_LSTM_MUL":
        core = LSTM2.LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_LSTM_MUL(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    opt.drop_prob_lm, opt.block_num)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_A":
        core = LSTM2.LSTM_DOUBLE_ATT_STACK_PARALLEL_A(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL":
        core = LSTM2.LSTM_SOFT_ATT_STACK_PARALLEL(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT":
        core = LSTM2.LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_WITH_MUL_WEIGHT":
        core = LSTM2.LSTM_SOFT_ATT_STACK_PARALLEL_WITH_MUL_WEIGHT(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_WEIGHT":
        core = LSTM2.LSTM_DOUBLE_ATT_STACK_PARALLEL_MUL_OUT_ATT_WITH_WEIGHT(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                    opt.num_parallels, opt.rnn_size, opt.att_size,
                                                    dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_SPP":
        core = LSTM3.LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_SPP(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                          opt.num_parallels, opt.rnn_size, opt.att_size,
                                                          opt.pool_size, opt.spp_num,
                                                          dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_SPP":
        core = LSTM3.LSTM_SOFT_ATT_STACK_PARALLEL_SPP(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                          opt.num_parallels, opt.rnn_size, opt.att_size,
                                                          opt.pool_size, opt.spp_num,
                                                          dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_MEMORY":
        core = LSTM4.LSTM_SOFT_ATT_STACK_PARALLEL_MEMORY(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                          opt.num_parallels, opt.rnn_size, opt.att_size, opt.memory_num_hop,
                                                          dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_NO_MEMORY":
        core = LSTM4.LSTM_SOFT_ATT_STACK_PARALLEL_NO_MEMORY(opt.input_encoding_size, opt.vocab_size + 1, opt.num_layers,
                                                          opt.num_parallels, opt.rnn_size, opt.att_size,
                                                          dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_BU":
        core = LSTM5.LSTM_SOFT_ATT_STACK_PARALLEL_WITH_WEIGHT_BU(opt.input_encoding_size, opt.vocab_size + 1,
                                                              opt.num_layers,
                                                              opt.num_parallels, opt.rnn_size, opt.att_size, opt.bu_size,
                                                              dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_C_S_ATT_STACK_PARALLEL_WITH_WEIGHT_BU":
        core = LSTM5.LSTM_C_S_ATT_STACK_PARALLEL_WITH_WEIGHT_BU(opt.input_encoding_size, opt.vocab_size + 1,
                                                              opt.num_layers,
                                                              opt.num_parallels, opt.rnn_size, opt.att_size, opt.bu_size,
                                                              dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_WITH_TOP_DOWN_ATTEN":
        core = LSTM6.LSTM_WITH_TOP_DOWN_ATTEN(opt.input_encoding_size, opt.vocab_size + 1,
                                                                 opt.num_layers,
                                                                 opt.num_parallels, opt.rnn_size, opt.att_size,
                                                                 opt.bu_size,
                                                                 opt.bu_num,
                                                                 dropout=opt.drop_prob_lm)
    elif opt.rnn_type == "LSTM_SOFT_ATT_STACK_PARALLEL_WITH_FC_WEIGHT":
        core = LSTM2.LSTM_SOFT_ATT_STACK_PARALLEL_WITH_FC_WEIGHT(opt.input_encoding_size, opt.vocab_size + 1,
                                                              opt.num_layers,
                                                              opt.num_parallels, opt.rnn_size, opt.att_size,
                                                              dropout=opt.drop_prob_lm)
    else:
        raise Exception("rnn type not supported: {}".format(opt.rnn_type))

    return core