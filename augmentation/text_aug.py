import os
import time
import nlpaug.augmenter.word as naw

from nlpaug.augmenter.word.back_translation import BackTranslationAug

load_start_time = time.time()
# WORD2VEC_MODEL_PATH = os.path.join(os.environ.get('EMBEDDINGS'),
#                                    'GoogleNews-vectors-negative300.bin')
# transformer_wmt19_en_de = os.path.join(os.environ.get('PRETRAINED'),
#                                        'transformer.wmt19.en-de')
# transformer_wmt19_de_en = os.path.join(os.environ.get('PRETRAINED'),
#                                        'transformer.wmt19.de-en')
WORD2VEC_MODEL = None
# WORD2VEC_MODEL = naw.WordEmbsAug.get_model(model_path=WORD2VEC_MODEL_PATH,
#                                            model_type='word2vec')
load_end_time = time.time()
print('model loaded cost', load_end_time - load_start_time, 's')


def Spelling(text, v):
    """ Substitute word by spelling mistake words dictionary
    """
    return naw.SpellingAug(aug_p=v).augment(text)


def WordEmbeddingsInsert(text, v):
    """ Insert word randomly by word embeddings similarity
    """
    return naw.WordEmbsAug(aug_p=v,
                           model=WORD2VEC_MODEL,
                           model_type='word2vec',
                           action="insert").augment(text)


def WordEmbeddingsSubstitute(text, v):
    """ Substitute word by word2vec similarity
    """
    return naw.WordEmbsAug(aug_p=v,
                           model=WORD2VEC_MODEL,
                           model_type='word2vec',
                           action="substitute").augment(text)


# def TFIDFInsert(text, aug_min, aug_max, aug_p):
#     """ Insert word by TF-IDF similarity
#     """
#     aug = naw.TfIdfAug(aug_min=aug_min,
#                        aug_max=aug_max,
#                        aug_p=aug_p,
#                        model_path='.',
#                        action="insert")
#     return aug.augment(text)

# def TFIDFSubstitute(text, aug_min, aug_max, aug_p):
#     """ Substitute word by TF-IDF similarity
#     """
#     aug = naw.TfIdfAug(aug_min=aug_min,
#                        aug_max=aug_max,
#                        aug_p=aug_p,
#                        model_path=os.environ.get("MODEL_DIR"),
#                        action="substitute")
#     return aug.augment(text)


def ContextualWordEmbeddingsInsert(text, v):
    """ Insert word by contextual word embeddings
    """
    return naw.ContextualWordEmbsAug(aug_p=v,
                                     device='cuda',
                                     model_path='bert-base-uncased',
                                     action="insert").augment(text)


def ContextualWordEmbeddingsSubstitute(text, v):
    """ Substitute word by contextual word embeddings
    """
    return naw.ContextualWordEmbsAug(aug_p=v,
                                     device='cuda',
                                     model_path='bert-base-uncased',
                                     action="substitute").augment(text)


def Synonym(text, v):
    """ Substitute word by WordNet's synonym
    """
    return naw.SynonymAug(aug_p=v, aug_src='wordnet').augment(text)


def Antonym(text, v):
    """ Substitute word by antonym
    """
    return naw.SynonymAug(aug_p=v).augment(text)


def RandomWordSwap(text, v):
    """ Swap word randomly
    """
    return naw.RandomWordAug(aug_p=v, action="swap").augment(text)


def RandomWordDelete(text, v):
    """ Delete word randomly
    """
    return naw.RandomWordAug(aug_p=v, action="delete").augment(text)


def RandomWordCrop(text, v):
    """ Delete a set of contunous word will be removed randomly
    """
    return naw.RandomWordAug(aug_p=v, action='crop').augment(text)


# def BackTranslation(text, v):
#     aug = naw.BackTranslationAug(from_model_name=transformer_wmt19_en_de,
#                                  to_model_name=transformer_wmt19_de_en,
#                                  is_load_from_github=False,
#                                  device='cuda')
#     return aug.augment(text)

text_transformations = [
    (Spelling, 0, 1),  # 1
    # (WordEmbeddingsInsert, 0, 1),  # 2
    # (WordEmbeddingsSubstitute, 0, 1),  # 3
    # (ContextualWordEmbeddingsInsert, 0, 1),  # 4
    # (ContextualWordEmbeddingsSubstitute, 0, 1),  # 5
    (Synonym, 0, 1),  # 6
    (Antonym, 0, 1),  # 7
    (RandomWordSwap, 0, 1),  # 8
    (RandomWordDelete, 0, 1),  # 9
    (RandomWordCrop, 0, 1),  # 10
    # (BackTranslation, 0, 1),  # 11
]

if __name__ == '__main__':
    text1 = 'The quick brown fox jumps over the lazy dog .'
    text2 = 'The authors claimed that within three or five years, machine translation ' \
        'would be a solved problem.'

    print('Origin text1:', text1)
    print('Origin text2:', text2)
    print('\n')

    start_time = time.time()
    auged_text1 = Spelling(text1, 0.3)
    auged_text2 = Spelling(text2, 0.3)
    end_time = time.time()
    print('Spelling text1:', auged_text1)
    print('Spelling text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = WordEmbeddingsInsert(text1, 0.3)
    auged_text2 = WordEmbeddingsInsert(text2, 0.3)
    end_time = time.time()
    print('WordEmbeddingsInsert text1:', auged_text1)
    print('WordEmbeddingsInsert text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = WordEmbeddingsSubstitute(text1, 0.3)
    auged_text2 = WordEmbeddingsSubstitute(text2, 0.3)
    end_time = time.time()
    print('WordEmbeddingsSubstitute text1:', auged_text1)
    print('WordEmbeddingsSubstitute text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = ContextualWordEmbeddingsInsert(text1, 0.3)
    auged_text2 = ContextualWordEmbeddingsInsert(text2, 0.3)
    end_time = time.time()
    print('ContextualWordEmbeddingsInsert text1:', auged_text1)
    print('ContextualWordEmbeddingsInsert text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = ContextualWordEmbeddingsSubstitute(text1, 0.3)
    auged_text2 = ContextualWordEmbeddingsSubstitute(text2, 0.3)
    end_time = time.time()
    print('ContextualWordEmbeddingsSubstitute text1:', auged_text1)
    print('ContextualWordEmbeddingsSubstitute text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = Synonym(text1, 0.3)
    auged_text2 = Synonym(text2, 0.3)
    end_time = time.time()
    print('Synonym text1:', auged_text1)
    print('Synonym text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = Antonym(text1, 0.3)
    auged_text2 = Antonym(text2, 0.3)
    end_time = time.time()
    print('Antonym text1:', auged_text1)
    print('Antonym text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = RandomWordSwap(text1, 0.3)
    auged_text2 = RandomWordSwap(text2, 0.3)
    end_time = time.time()
    print('RandomWordSwap text1:', auged_text1)
    print('RandomWordSwap text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = RandomWordDelete(text1, 0.3)
    auged_text2 = RandomWordDelete(text2, 0.3)
    end_time = time.time()
    print('RandomWordDelete text1:', auged_text1)
    print('RandomWordDelete text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = RandomWordCrop(text1, 0.3)
    auged_text2 = RandomWordCrop(text2, 0.3)
    end_time = time.time()
    print('RandomWordCrop text1:', auged_text1)
    print('RandomWordCrop text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')

    start_time = time.time()
    auged_text1 = BackTranslation(text1, 0.3)
    auged_text2 = BackTranslation(text2, 0.3)
    end_time = time.time()
    print('BackTranslation text1:', auged_text1)
    print('BackTranslation text2:', auged_text2)
    print(f'Cost: {end_time - start_time}\n')
