import torch
from transformers import BertTokenizer
from data_transform_overlapping import overlapping_long_texts, encode_labels
from bert_ml_sentiment_classifier import calc_norm

import pandas as pd

def overlapping_long_texts_test():
    print("I'm ok.")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    print("Reading data...")
    df_data = pd.read_csv("/home/andrazp/cross-lingual_sentiment/cross-lingual_sentiment/data/Sentinews_document_prepared_full.tsv", sep="\t")
    data = df_data['data'].tolist()
    # print(df_data['label'].tolist())
    #label_set = list(set(df_data['label'].values))
    #labels = encode_labels(df_data['label'].tolist(), label_set)

    for sentence in data:
        overlapping_long_texts(sentence, tokenizer, 512)

    print("Done")


def test_calc_norm():
    list_of_vectors = []
    x = torch.rand(1, 5)
    y = torch.rand(1, 5)
    print(x)
    print(y)
    list_of_vectors.append(x)
    list_of_vectors.append(y)
    result = calc_norm(list_of_vectors)
    print(result)


if __name__ == "__main__":
    test_calc_norm()
