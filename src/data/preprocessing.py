import pickle
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.config import BINARY_VOCAB_PATH, MAX_LEN


def clean_word(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z ]', '', sentence)
    return sentence


def read_data(filepath):
    df = pd.read_csv(filepath)
    similarity_map = {
        'neutral': np.nan,
        'contradiction': 0,
        'entailment': 1,
        '-': np.nan
    }
    df['similarity'] = df['similarity'].apply(
        lambda column: similarity_map[column])
    df.dropna(axis=0, inplace=True)
    df['sentence1'] = df['sentence1'].apply(clean_word)
    df['sentence2'] = df['sentence2'].apply(clean_word)
    return df


def tokenize(sentences):
    tokens = []
    for sentence in sentences:
        tokens.append(word_tokenize(sentence))
    return tokens


def stem_word(sentences):
    tokens = []
    stemmer = PorterStemmer()
    for sentence in sentences:
        sentence_tokens = []
        for word in sentence:
            sentence_tokens.append(stemmer.stem(word))
        tokens.append(sentence_tokens)

    return tokens


def remove_stopwords(sentences):
    tokens = []
    stop = stopwords.words('english')
    for sentence in sentences:
        new_sen = []
        for word in sentence:
            if word not in stop:
                new_sen.append(word)
        tokens.append(new_sen)
    return tokens


def flatten(item_list):
    return [item for sublist in item_list for item in sublist]


def create_vocabulary(sentence1, sentence2):
    sentence1 = set(flatten(sentence1))
    sentence2 = set(flatten(sentence2))
    vocab = sentence1.union(sentence2)
    return sorted(list(vocab))


def create_mappings(vocab):
    word2idx = {word: idx + 2 for idx, word in enumerate(vocab)}
    idx2word = {idx + 2: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def map_to_token(map_dict, tokens):
    all_tokens = []
    for sentence in tokens:
        sentence_tokens = []
        for word in sentence:
            if word in map_dict.keys():
                sentence_tokens.append(map_dict[word])
            else:
                sentence_tokens.append(1)
        all_tokens.append(sentence_tokens)
    return all_tokens


def prepare_data(
        sentence1,
        sentence2,
        vocab_path,
        training=True,
        return_vocab=False):
    tokens1 = tokenize(sentence1)
    tokens2 = tokenize(sentence2)

    stemed_tokens1 = stem_word(tokens1)
    stemed_tokens2 = stem_word(tokens2)

    if training:
        vocab = create_vocabulary(
            stemed_tokens1, stemed_tokens2)
        pickle.dump(vocab, open(vocab_path, 'wb'))

    else:
        vocab = pickle.load(open(vocab_path, 'rb'))

    word2idx, idx2word = create_mappings(vocab)

    idx_tokens1 = map_to_token(word2idx, stemed_tokens1)
    idx_tokens2 = map_to_token(word2idx, stemed_tokens2)

    padded_tokens1 = pad_sequences(idx_tokens1, maxlen=MAX_LEN)
    padded_tokens2 = pad_sequences(idx_tokens2, maxlen=MAX_LEN)

    data1 = np.array(padded_tokens1, dtype='object').astype('int32')
    data2 = np.array(padded_tokens2, dtype='object').astype('int32')
    if return_vocab:
        return vocab, data1, data2
    return data1, data2


def return_vocab(vocab_path):
    vocab = pickle.load(open(vocab_path, 'rb'))
    return vocab
