from autoaugment.domain.nlp.classification.models.HAN.han import HAN
from autoaugment.domain.nlp.classification.models.fastText.fasttext import fastText
from autoaugment.domain.nlp.classification.models.AttBiLSTM.att_bilstm import AttBiLSTM
from autoaugment.domain.nlp.classification.models.TextCNN.cnn1d import TextCNN1D
from autoaugment.domain.nlp.classification.models.TextCNN.cnn2d import TextCNN2D
from autoaugment.domain.nlp.classification.models.Transformer.transformer import Transformer
'''
setup a model

input params:
    config (Class): config settings
    n_classes: number of classes
    vocab_size: size of vocabulary
    embeddings: word embeddings
    emb_size: size of word embeddings
'''


def setup(config, n_classes, vocab_size, embeddings, emb_size):

    if config['model_name'] == 'han':
        model = HAN(n_classes=n_classes,
                    vocab_size=vocab_size,
                    embeddings=embeddings,
                    emb_size=emb_size,
                    fine_tune=config['fine_tune_word_embeddings'],
                    word_rnn_size=config['word_rnn_size'],
                    sentence_rnn_size=config['sentence_rnn_size'],
                    word_rnn_layers=config['word_rnn_layers'],
                    sentence_rnn_layers=config['sentence_rnn_layers'],
                    word_att_size=config['word_att_size'],
                    sentence_att_size=config['sentence_att_size'],
                    dropout=config['dropout'])
    elif config['model_name'] == 'fasttext':
        model = fastText(n_classes=n_classes,
                         vocab_size=vocab_size,
                         embeddings=embeddings,
                         emb_size=emb_size,
                         fine_tune=config['fine_tune_word_embeddings'],
                         hidden_size=config['hidden_size'])
    elif config['model_name'] == 'attbilstm':
        model = AttBiLSTM(n_classes=n_classes,
                          vocab_size=vocab_size,
                          embeddings=embeddings,
                          emb_size=emb_size,
                          fine_tune=config['fine_tune_word_embeddings'],
                          rnn_size=config['rnn_size'],
                          rnn_layers=config['rnn_layers'],
                          dropout=config['dropout'])
    elif config['model_name'] == 'textcnn':
        if config['conv_layer'] == '2D':
            model = TextCNN2D(n_classes=n_classes,
                              vocab_size=vocab_size,
                              embeddings=embeddings,
                              emb_size=emb_size,
                              fine_tune=config['fine_tune_word_embeddings'],
                              n_kernels=config['n_kernels'],
                              kernel_sizes=config['kernel_sizes'],
                              n_channels=config['n_channels'],
                              dropout=config['dropout'])
        elif config['conv_layer'] == '1D':
            model = TextCNN1D(n_classes=n_classes,
                              vocab_size=vocab_size,
                              embeddings=embeddings,
                              emb_size=emb_size,
                              fine_tune=config['fine_tune_word_embeddings'],
                              n_kernels=config['n_kernels'],
                              kernel_sizes=config['kernel_sizes'],
                              n_channels=config['n_channels'],
                              dropout=config['dropout'])
        else:
            raise Exception("Convolution layer not supported: ",
                            config['conv_layer'])
    elif config['model_name'] == 'transformer':
        model = Transformer(n_classes=n_classes,
                            vocab_size=vocab_size,
                            embeddings=embeddings,
                            d_model=emb_size,
                            word_pad_len=config['word_limit'],
                            fine_tune=config['fine_tune_word_embeddings'],
                            hidden_size=config['hidden_size'],
                            n_heads=config['n_heads'],
                            n_encoders=config['n_encoders'],
                            dropout=config['dropout'])
    else:
        raise Exception("Model not supported: ", config['model_name'])

    return model
