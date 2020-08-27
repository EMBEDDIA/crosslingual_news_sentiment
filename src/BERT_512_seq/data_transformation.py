import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from keras_preprocessing.sequence import pad_sequences

import math
import sys

def encode_labels(labels, labels_set):
    """Maps each label to a unique index.
    :param labels: (list of strings) labels of every instance in the dataset
    :param labels_set: (list of strings) set of labels that appear in the dataset
    :return (list of int) encoded labels
    """
    encoded_labels = []
    for label in labels:
        encoded_labels.append(labels_set.index(label))
    return encoded_labels


def cut_at_length(data, labels, tokenizer, max_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    #sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in data]
    truncated_sentences = [sentence[:(max_len - 2)] for sentence in tokenized_sentences]
    truncated_sentences = [["[CLS]"] + sentence + ["[SEP]"] for sentence in truncated_sentences]
    print("Example of tokenized sentence:")
    print(truncated_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in truncated_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def cut_at_front_and_back(data, labels, tokenizer, max_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    sentences = ["[CLS] " + sentence for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

    cut_tokenized_sentences = []
    for tokenized_sentence in tokenized_sentences:
        if len(tokenized_sentence) < max_len:
            cut_tokenized_sentences.append(tokenized_sentence + ["[SEP]"])
        elif len(tokenized_sentence) > max_len:
            tokenized_sentence = tokenized_sentence[:math.floor(max_len / 2)] + \
                           tokenized_sentence[-(math.ceil(max_len / 2) - 1):] + ["[SEP]"]
            cut_tokenized_sentences.append(tokenized_sentence)
        else:
            tokenized_sentence = tokenized_sentence[:-1] + ["[SEP]"]
            cut_tokenized_sentences.append(tokenized_sentence)

    print("Example of tokenized sentence:")
    print(cut_tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in cut_tokenized_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])

    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    #print("Printing cut sequence:")
    #print(cut_input_ids[0])
    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader
