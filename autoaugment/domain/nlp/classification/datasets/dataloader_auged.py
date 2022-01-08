'''
load data from manually preprocessed data (see datasets/prepocess/)
'''
import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.tokenize import TreebankWordTokenizer
from autoaugment.domain.nlp.classification.datasets.preprocess.sentence import get_clean_text
from autoaugment.domain.nlp.classification.datasets.info import get_label_map
from autoaugment.domain.nlp.classification.utils.embedding import load_embeddings
from autoaugment.domain.nlp.classification.datasets.preprocess.sentence import encode_and_pad
'''
a PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
(for document classification)

attributes:
    data_folder: folder where data files are stored
    split: split, one of 'TRAIN' or 'TEST'
'''


class DocDataset(Dataset):
    def __init__(self, data_folder, split):
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(
            os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self):
        return len(self.data['labels'])


'''
a PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
(for sentence classification)

attributes:
    data_folder: folder where data files are stored
    split: split, one of 'TRAIN' or 'TEST'
'''

# tokenizers
word_tokenizer = TreebankWordTokenizer()


def read_csv(config, split, train_size=0.1):
    texts = []
    labels = []
    path = os.path.join(config['dataset_path'], split.lower() + '.csv')
    df = pd.read_csv(path, header=None)
    for i in range(df.shape[0]):
        row = list(df.loc[i, :])
        s = ''
        for text in row[1:]:
            text = get_clean_text(text)
            s += text
        texts.append(s)
        labels.append(int(row[0]) - 1)
    if split.upper() == 'TEST' or (split.upper() == 'TRAIN'
                                   and train_size == 1):
        return {'texts': texts, 'labels': labels}
    X_train, _, y_train, _ = train_test_split(texts,
                                              labels,
                                              train_size=train_size,
                                              stratify=labels,
                                              random_state=42)
    return {'texts': X_train, 'labels': y_train}


class SentDataset(Dataset):
    def __init__(self, config, split):
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split
        # 加载CSV文件
        self.data = read_csv(config,
                             split,
                             train_size=config['reduced_train_size'])

    def __getitem__(self, i):
        return self.data['texts'][i], None, self.data['labels'][i]

    def __len__(self):
        return len(self.data['labels'])


'''
load data from files output by prepocess.py

input param:
    config (Class): config settings
    split: 'trian' / 'test'
    build_vocab: build vocabulary?
                 only makes sense when split = 'train'

return:
    split = 'test':
        test_loader: dataloader for test data
    split = 'train':
        build_vocab = Flase:
            train_loader: dataloader for train data
        build_vocab = True:
            train_loader: dataloader for train data
            embeddings: pre-trained word embeddings (None if config.emb_pretrain = False)
            emb_size: embedding size (config.emb_size if config.emb_pretrain = False)
            word_map: word2ix map
            n_classes: number of classes
            vocab_size: size of vocabulary
'''


def load_data(config,
              split,
              build_vocab=True,
              augmentation_func=None,
              custom_aug=None,
              augmentation=None):
    split = split.lower()
    assert split in {'train', 'test'}

    # load word2ix map
    with open(os.path.join(config['output_path'], 'word_map.json'), 'r') as j:
        word_map = json.load(j)

    # 针对每个 batch 做数据增强处理
    def augment_batch_text(batch):
        if (augmentation_func is None or custom_aug is None
                or len(custom_aug) == 0) and augmentation is None:
            return test_collate_fn(batch)
        texts, _, labels = list(zip(*batch))
        auged_texts = []
        for t in texts:
            # token_text = word_tokenizer.tokenize(t)
            # auged_texts.append(token_text[:config['word_limit']])
            if augmentation_func is not None:
                auged_texts.append(
                    word_tokenizer.tokenize(augmentation_func(custom_aug)(t))
                    [:config['word_limit']])
            elif augmentation is not None:
                auged_texts.append(
                    word_tokenizer.tokenize(
                        augmentation(t))[:config['word_limit']])
        encoded_train_sents, words_per_train_sent = encode_and_pad(
            auged_texts, word_map, config['word_limit'])
        sentences = torch.LongTensor(encoded_train_sents)
        words_per_sentence = torch.LongTensor(words_per_train_sent)
        labels = torch.LongTensor(labels)
        return sentences, words_per_sentence, labels

    def test_collate_fn(batch):
        texts, _, labels = list(zip(*batch))
        limited_texts = []
        for t in texts:
            token_text = word_tokenizer.tokenize(t)
            limited_texts.append(token_text[:config['word_limit']])
        encoded_train_sents, words_per_train_sent = encode_and_pad(
            limited_texts, word_map, config['word_limit'])
        sentences = torch.LongTensor(encoded_train_sents)
        words_per_sentence = torch.LongTensor(words_per_train_sent)
        labels = torch.LongTensor(labels)
        return sentences, words_per_sentence, labels

    # test
    if split == 'test':
        test_loader = DataLoader(
            DocDataset(config['output_path'], 'test') if config['model_name']
            in ['han'] else SentDataset(config, 'test'),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['workers'],
            pin_memory=True,
            collate_fn=test_collate_fn)
        return test_loader
    # train
    else:
        # dataloaders
        train_loader = DataLoader(
            DocDataset(config['output_path'], 'train') if config['model_name']
            in ['han'] else SentDataset(config, 'train'),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['workers'],
            pin_memory=True,
            collate_fn=augment_batch_text)

        if not build_vocab:
            return train_loader

        else:
            # size of vocabulary
            vocab_size = len(word_map)

            # number of classes
            label_map, _ = get_label_map(config['dataset'])
            n_classes = len(label_map)

            # word embeddings
            if config['emb_pretrain']:
                # load Glove as pre-trained word embeddings for words in the word map
                # emb_path = os.path.join(config.emb_folder, config.emb_filename)
                embeddings, emb_size = load_embeddings(
                    emb_file=os.path.join(config['emb_folder'],
                                          config['emb_filename']),
                    word_map=word_map,
                    output_folder=config['output_path'])
            # or initialize embedding weights randomly
            else:
                embeddings = None
                emb_size = config['emb_size']

            return train_loader, embeddings, emb_size, word_map, n_classes, vocab_size


if __name__ == '__main__':
    import pickle
    from theconf import Config as C, ConfigArgumentParser
    parser = ConfigArgumentParser(conflict_handler='resolve')
    args = parser.parse_args()
    config = C.get()
    from autoaugment.augmentation.augmentation import Augmentation
    from autoaugment.augmentation.text_aug import text_transformations
    from autoaugment.search.common.utils import generate_subpolicies
    augmentation_func = Augmentation
    w_policy = pickle.load(
        open(
            'experiments/sges_exp0317_00_wrn_40x2_100_100_0.2/last_line_policy.pickle',
            mode='rb'))
    sub_policies = generate_subpolicies(w_policy,
                                        denormalize=True,
                                        augmentation_list=text_transformations)
    train_loader = load_data(config=config,
                             split='train',
                             build_vocab=False,
                             augmentation_func=augmentation_func,
                             custom_aug=sub_policies)
    batch = next(iter(train_loader))
