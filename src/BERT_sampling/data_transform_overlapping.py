from bert_ml_sentiment_classifier import bert_extract_CLS_embedding

import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from keras_preprocessing.sequence import pad_sequences


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


def overlapping_long_texts(sentence, tokenizer, max_len):
    tokenized_sentence = tokenizer.tokenize(sentence)

    tokenized_sequences = []
    if len(tokenized_sentence) <= (max_len - 2):
        tokenized_sentence.insert(0, "[CLS]")
        tokenized_sentence.append("[SEP]")
        tokenized_sequences.append(tokenized_sentence)
    elif len(tokenized_sentence) > (max_len - 2):
        s_slice = 0
        while True:
            if (s_slice + max_len - 2) > len(tokenized_sentence):
                final_sequence = tokenized_sentence[s_slice:]
                final_sequence.insert(0, "[CLS]")
                final_sequence.append("[SEP]")
                tokenized_sequences.append(final_sequence)
                break
            final_sequence = tokenized_sentence[s_slice:(s_slice + max_len - 2)]
            final_sequence.insert(0, "[CLS]")
            final_sequence.append("[SEP]")
            tokenized_sequences.append(final_sequence)
            s_slice = s_slice + max_len - 52  # to offset the -2 because of two additional tokens (CLS and SEP)

    return tokenized_sequences


def prepare_labeled_dataset(data, labels, tokenizer, max_len, batch_size):
    oversampled_data = []
    oversampled_labels = []
    for seq, label in zip(data, labels):
        tokenized_sequences = overlapping_long_texts(seq, tokenizer, max_len)
        for tok_seq in tokenized_sequences:
            oversampled_data.append(tok_seq)
            oversampled_labels.append(label)

    print("Example of tokenized sentence:")
    print(oversampled_data[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in oversampled_data]
    print("Printing encoded sentences:")
    print(input_ids[0])

    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # print("Printing cut sequence:")
    # print(cut_input_ids[0])
    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(oversampled_labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def prepare_prediction_dataset(data, tokenizer, max_len, batch_size):
    oversampled_data = []
    for seq in data:
        tokenized_sentence = overlapping_long_texts(seq, tokenizer, max_len)
        for tok_seq in tokenized_sentence:
            oversampled_data.append(tok_seq)

    print("Example of tokenized sentence:")
    print(oversampled_data[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in oversampled_data]
    print("Printing encoded sentences:")
    print(input_ids[0])

    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # print("Printing cut sequence:")
    # print(cut_input_ids[0])
    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks)
    sampler = SequentialSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def prepare_classification_head_dataset(train_data, train_labels, model, tokenizer, device, max_len, batch_size):
    """Needs a BERT model"""
    docs = []
    for seq in train_data:
        #batch_size is 1 so that bert_predict returns the last layer for each instance seperately
        seq = [seq]  # so that we have a list of strings
        seq_dataloader = prepare_prediction_dataset(seq, tokenizer, max_len, 1)
        cls_embeddings = bert_extract_CLS_embedding(model, seq_dataloader, device)
        final_embedding = calc_norm(cls_embeddings)
        docs.append(final_embedding)
    print(docs)
    print(type(docs))
    docs = torch.stack(docs)
    train_labels = torch.tensor(train_labels)
    doc_data = TensorDataset(docs, train_labels)
    sampler = SequentialSampler(doc_data)
    dataloader = DataLoader(doc_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def prepare_classification_head_dataset_averaging(train_data, train_labels, model, tokenizer, device, max_len, batch_size):
    """Needs a BERT model"""
    docs = []
    for seq in train_data:
        #batch_size is 1 so that bert_predict returns the last layer for each instance seperately
        seq = [seq]  # so that we have a list of strings
        seq_dataloader = prepare_prediction_dataset(seq, tokenizer, max_len, 1)
        cls_embeddings = bert_extract_CLS_embedding(model, seq_dataloader, device)
        cls_embeddings = torch.stack(cls_embeddings)
        final_embedding = cls_embeddings.mean(0)
        print(final_embedding.size())
        docs.append(final_embedding)
    print(docs)
    print(type(docs))
    docs = torch.stack(docs)
    train_labels = torch.tensor(train_labels)
    doc_data = TensorDataset(docs, train_labels)
    sampler = SequentialSampler(doc_data)
    dataloader = DataLoader(doc_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def prepare_classification_head_CNN_dataset(train_data, train_labels, model, tokenizer, device, max_len, batch_size):
    """Needs a BERT model"""
    docs = []
    for seq in train_data:
        #batch_size is 1 so that bert_predict returns the last layer for each instance seperately
        seq = [seq]  # so that we have a list of strings
        seq_dataloader = prepare_prediction_dataset(seq, tokenizer, max_len, 1)
        cls_embeddings = bert_extract_CLS_embedding(model, seq_dataloader, device)
        cls_embeddings = pad_sequence_for_cnn(cls_embeddings)
        docs.append(cls_embeddings)
    print(docs)
    print(type(docs))
    docs = torch.stack(docs)
    train_labels = torch.tensor(train_labels)
    doc_data = TensorDataset(docs, train_labels)
    sampler = SequentialSampler(doc_data)
    dataloader = DataLoader(doc_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def pad_sequence_for_cnn(cls_embeddings):
    if len(cls_embeddings) > 6:
        cls_embeddings = cls_embeddings[:6]
    elif len(cls_embeddings) < 6:
        while len(cls_embeddings) < 6:
            zero_tensor = torch.zeros_like(cls_embeddings[0])
            cls_embeddings.append(zero_tensor)
    cls_embeddings = torch.stack(cls_embeddings)
    cls_embeddings = torch.transpose(cls_embeddings, 0, 1)
    #cls_embeddings = pack_padded_sequences(cls_embeddings, 6, batch_first=True, enforce_sorted=False)
    return cls_embeddings


def prepare_classification_head_weighted_dataset(train_data, train_labels, model, tokenizer, device, max_len, batch_size):
    """Needs a BERT model"""
    docs = []
    for seq in train_data:
        #batch_size is 1 so that bert_predict returns the last layer for each instance seperately
        seq = [seq]  # so that we have a list of strings
        seq_dataloader = prepare_prediction_dataset(seq, tokenizer, max_len, 1)
        cls_embeddings = bert_extract_CLS_embedding(model, seq_dataloader, device)
        cls_embeddings = pad_sequence_for_weighted(cls_embeddings)
        docs.append(cls_embeddings)
    print(docs)
    print(type(docs))
    docs = torch.stack(docs)
    train_labels = torch.tensor(train_labels)
    doc_data = TensorDataset(docs, train_labels)
    sampler = SequentialSampler(doc_data)
    dataloader = DataLoader(doc_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def pad_sequence_for_weighted(cls_embeddings):
    if len(cls_embeddings) > 6:
        cls_embeddings = cls_embeddings[:6]
    elif len(cls_embeddings) < 6:
        while len(cls_embeddings) < 6:
            zero_tensor = torch.zeros_like(cls_embeddings[0])
            cls_embeddings.append(zero_tensor)
    cls_embeddings = torch.stack(cls_embeddings)
    return cls_embeddings


def calc_norm(x):
    vector = None
    norm = float('-inf')
    for v in x:
        v_norm = torch.norm(v, p=2)
        v_norm = v_norm.item()
        if v_norm > norm:
            norm = v_norm
            vector = v
    return vector
