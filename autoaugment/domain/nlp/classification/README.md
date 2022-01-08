# Text Classification

PyTorch re-implementation of some text classificaiton models.

&nbsp;
## Model List

You can train following models by configuring `model_name` in config files ([here](https://github.com/Renovamen/Text-Classification/tree/master/configs) are some example config files). Check out their links for more info.

- [**Hierarchical Attention Networks (HAN)**](https://github.com/Renovamen/Text-Classification/tree/master/models/HAN) (`han`)

    **Hierarchical Attention Networks for Document Classification.** *Zichao Yang, et al.* NAACL 2016. [[Paper]](https://www.aclweb.org/anthology/N16-1174.pdf)

- [**fastText**](https://github.com/Renovamen/Text-Classification/tree/master/models/fastText) (`fasttext`)

    **Bag of Tricks for Efficient Text Classification.** *Armand Joulin, et al.* EACL 2017. [[Paper]](https://www.aclweb.org/anthology/E17-2068.pdf) [[Code]](https://github.com/facebookresearch/fastText)

- [**Bi-LSTM + Attention**](https://github.com/Renovamen/Text-Classification/tree/master/models/AttBiLSTM) (`attbilstm`)

    **Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.** *Peng Zhou, et al.* ACL 2016. [[Paper]](https://www.aclweb.org/anthology/P16-2034.pdf)

- [**TextCNN**](https://github.com/Renovamen/Text-Classification/tree/master/models/TextCNN) (`textcnn`)

    **Convolutional Neural Networks for Sentence Classification.** *Yoon Kim.* EMNLP 2014. [[Paper]](https://www.aclweb.org/anthology/D14-1181.pdf) [[Code]](https://github.com/yoonkim/CNN_sentence)

- [**Transformer**](https://github.com/Renovamen/Text-Classification/tree/master/models/Transformer) (`transformer`)

    **Attention Is All You Need.** *Ashish Vaswani, et al.* NIPS 2017. [[Paper]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[Code]](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
    
    Here only the encoder part of Transformer is used.

&nbsp;
## Environment

- Python 3.7

- [PyTorch](https://pytorch.org/) 1.5.0

- [Tensorflow](https://www.tensorflow.org/) 2.0.0 (optional, you don't need this if you disable [tensorboard](https://github.com/tensorflow/tensorboard))


&nbsp;
## Dataset

Currently, the following datasets proposed in [this paper](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) are supported:

- AG News   

- DBpedia

- Yelp Review Polarity

- Yelp Review Full

- Yahoo Answers

- Amazon Review Full

- Amazon Review Polarity

And all of them can be download [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) (Google Drive). Check out [here](docs/datasets.md) for more info about these datasets.

You should download and unzip them first, then set their path (`dataset_path`) in your config files. If you would like to use other datasets, they may have to be stored in the same format as the above mentioned datasets.

&nbsp;
## Pre-trained Word Embeddings

If you would like to use pre-trained word embeddings (like [GloVe](https://github.com/stanfordnlp/GloVe)), just set `emb_pretrain = True` and the path to the pre-trained vectors file (`emb_folder` and `emb_filename`) in your config files. You could also choose to fine-tune or not with the `fine_tune_embeddings` parameter.

The `load_embeddings` method (in [`utils/embedding.py`](utils/embedding.py)) would create a cache under folder `dataset_output_path`, so that it could load the embeddings quicker the next time.

Or if you want to randomly initialize the embedding layer's weights, set `emb_pretrain = False` and specify the size of embedding layer (`embed_size`).


&nbsp;
## Preprocess

Although [torchtext](https://github.com/pytorch/text) can be used to perform data preprocessing easily, it loads all data in one go and occupies too much memory and slows down the training speed, expecially when the dataset is big. 

Therefore, here I preprocess the data manually and store them locally first (where `configs/test.yaml` is the path to your config file):

```bash
python preprocess.py --config configs/example.yaml 
```

Then I load data dynamically using PyTorch's Dataloader when training (see [`datasets/dataloader.py`](datasets/dataloader.py)).

The preprocessing including encoding and padding sentences and building word2ix map. This may takes a little time, but in this way, the training can occupy less memory (which means we can have a large batch size) and take less time. For example, I need 4.6 minutes (on RTX 2080 Ti) to train a fastText model on Yahoo Answers dataset for an epoch using torchtext, but only 41 seconds using Dataloader.

[`torchtext.py`](https://github.com/Renovamen/Text-Classification/blob/abandoned/datasets/torchtext.py) is the script for loading data via torchtext, you can try it if you have interests.


&nbsp;
## Train

To train a model, just run:

```bash
python train.py --config configs/example.yaml
```

If you have enabled the tensorboard (`tensorboard: True` in config files), you can visualize the losses and accuracies during training by:

```bash
tensorboard --logdir=<your_log_dir>
```

&nbsp;

## Test

Test the trained model and compute accuracy on test set:

```bash
python test.py --config configs/example.yaml
```

&nbsp;
## Classify

This is for when you have already trained a model and want to predict a category for a specific sentence:

First modify following things in [`classify.py`](classify.py):

```python
checkpoint_path = 'str: path_to_your_checkpoint'

# pad limits
# only makes sense when model_name == 'han'
sentence_limit_per_doc = 15
word_limit_per_sentence = 20
# only makes sense when model_name != 'han'
word_limit = 200
```

Then, run:

```bash
python classify.py
```

&nbsp;
## Performance

Here I report the test accuracy (%) and training time per epoch (on RTX 2080 Ti) of each model on various datasets. Model parameters are not carefully tuned, so better performance can be achieved by some parameter tuning.

|                            Model                             |  AG News   |   DBpedia   | Yahoo Answers |
| :----------------------------------------------------------: | :--------: | :---------: | :-----------: |
| [Hierarchical Attention Network](https://github.com/Renovamen/Text-Classification/tree/master/models/HAN) | 92.7 (45s) | 98.2 (70s)  |  74.5 (2.7m)  |
| [fastText](https://github.com/Renovamen/Text-Classification/tree/master/models/fastText) | 91.6 (8s)  | 97.9 (25s)  |  66.7 (41s)   |
| [Bi-LSTM + Attention ](https://github.com/Renovamen/Text-Classification/tree/master/models/AttBiLSTM) | 92.0 (50s) | 99.0 (105s) |  73.5 (3.4m)  |
| [TextCNN ](https://github.com/Renovamen/Text-Classification/tree/master/models/TextCNN) | 92.2 (24s) | 98.5 (100s) |   72.8 (4m)   |
| [Transformer](https://github.com/Renovamen/Text-Classification/tree/master/models/Transformer) | 92.2 (60s) | 98.6 (8.2m) |  72.5 (14.5m)  |


&nbsp;
## Acknowledgement

This project is based on [sgrvinod/a-PyTorch-Tutorial-to-Text-Classification](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification), thanks for this great work.