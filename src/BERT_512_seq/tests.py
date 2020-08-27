import pandas as pd
import math
from transformers import BertTokenizer


def test_label_set():
    df_data = pd.read_csv("../data/Sentinews_document_prepared_full.tsv", sep="\t")
    #data = df_data['data'].tolist()
    # print(df_data['label'].tolist())
    label_set = list(set(df_data['label'].values))
    label_set = sorted(label_set)
    print(label_set)


def test_cut_front_and_back():
    lists = [['a', 'b'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c']]
    cut_lists = []
    max_len = 3
    for l in lists:
        if len(l) < max_len:
            cut_lists.append(l + ["[SEP]"])
        elif len(l) > max_len:
            l = l[:math.floor(max_len / 2)] + \
                           l[-(math.ceil(max_len / 2) - 1):] + ["[SEP]"]
            cut_lists.append(l)
        else:
            l = l[:-1] + ["[SEP]"]
            cut_lists.append(l)

    print(cut_lists)


def test_corssvalidation_split():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in range(10):
        test_data = data[(math.floor(len(data) * i * 0.1)):(math.floor(len(data) * (i + 1) * 0.1))]
        test_labels = labels[math.floor((len(labels) * i * 0.1)):math.floor((len(labels) * (i + 1) * 0.1))]
        train_data = data[:math.floor((len(data) * i * 0.1))] + data[math.floor((len(data) * (i + 1) * 0.1)):]
        train_labels = labels[:math.floor((len(labels) * i * 0.1))] + labels[math.floor((len(labels) * (i + 1) * 0.1)):]
        print("Iteration {}".format(i))
        print(test_data)
        print(test_labels)
        print(train_data)
        print(train_labels)


def test_cut_at_front_and_back():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    print("Tokenizer loaded")
    df_data = pd.read_csv("../data/Sentinews_document_prepared_full.tsv", sep="\t")
    data = df_data['data'].tolist()
    max_len = 512

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

    for sent in cut_tokenized_sentences:
        if sent[-1] is not "[SEP]":
            print(sent)


if __name__ == "__main__":
    test_cut_at_front_and_back()
