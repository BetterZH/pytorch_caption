

def get_transformer_decoder(opt):

    print('transformer_decoder_type: ' + opt.transformer_decoder_type)

    # decoder
    if opt.transformer_decoder_type == "ImageDecoder":
        import transformer.TransformerDecoder as TransformerDecoder
        decoder = TransformerDecoder.ImageDecoder(opt)
    elif opt.transformer_decoder_type == "ImageAttentionDecoder":
        import transformer.TransformerAttention as TransformerAttention
        decoder = TransformerAttention.ImageAttentionDecoder(opt)
    elif opt.transformer_decoder_type == "ImageMoreSupDecoder":
        import transformer.TransformerMoreSupDecoder as TransformerMoreSupDecoder
        decoder = TransformerMoreSupDecoder.ImageMoreSupDecoder(opt)
    elif opt.transformer_decoder_type == "ImageMoreSupAttentionDecoder":
        import transformer.TransformerMoreSupDecoder as TransformerMoreSupDecoder
        decoder = TransformerMoreSupDecoder.ImageMoreSupAttentionDecoder(opt)
    elif opt.transformer_decoder_type == "ImageCDecoder":
        import transformer.TransformerDecoder as TransformerDecoder
        decoder = TransformerDecoder.ImageCDecoder(opt)
    elif opt.transformer_decoder_type == "ImageCMDecoder":
        import transformer.TransformerDecoder as TransformerDecoder
        decoder = TransformerDecoder.ImageCMDecoder(opt)
    elif opt.transformer_decoder_type == "ImageCMWGDecoder":
        import transformer.TransformerDecoder as TransformerDecoder
        decoder = TransformerDecoder.ImageCMWGDecoder(opt)
    elif opt.transformer_decoder_type == "ImageCMSDecoder":
        import transformer.TransformerMoreSupDecoder as TransformerMoreSupDecoder
        decoder = TransformerMoreSupDecoder.ImageCMSDecoder(opt)
    elif opt.transformer_decoder_type == "ImageCRMSDecoder":
        import transformer.TransformerMoreSupDecoder as TransformerMoreSupDecoder
        decoder = TransformerMoreSupDecoder.ImageCRMSDecoder(opt)
    else:
        raise Exception("transformer decoder type not supported: {}".format(opt.transformer_decoder_type))

    return decoder

def get_transformer_decoder_layer(opt):

    # decoder
    if opt.transformer_decoder_layer_type == "DecoderLayer":
        import transformer.TransformerDecoderLayer as TransformerDecoderLayer
        decoder_layer = TransformerDecoderLayer.DecoderLayer(opt)
    elif opt.transformer_decoder_layer_type == "AttentionDecoderLayer":
        import transformer.TransformerDecoderLayer as TransformerDecoderLayer
        decoder_layer = TransformerDecoderLayer.AttentionDecoderLayer(opt)
    elif opt.transformer_decoder_layer_type == "CDecoderLayer":
        import transformer.TransformerDecoderLayer as TransformerDecoderLayer
        decoder_layer = TransformerDecoderLayer.CDecoderLayer(opt)
    else:
        raise Exception("transformer decoder layer type not supported: {}".format(opt.transformer_decoder_layer_type))

    return decoder_layer