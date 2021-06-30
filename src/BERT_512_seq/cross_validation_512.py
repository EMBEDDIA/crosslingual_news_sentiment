from bert_ml_sentiment_classifier import bert_train, bert_evaluate
from data_transformation import encode_labels, cut_at_length

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np

import random
import argparse
import os
import csv
from math import floor

def length_512_crossvalidation():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--cro_test_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir",
                        required=True,
                        type=str)
    parser.add_argument("--eval_split",
                        default=0.2,
                        type=float)
    parser.add_argument("--test_split",
                        default=0.1,
                        type=float)
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--batch_size",
                        default=32,
                        type=int)
    parser.add_argument("--num_epochs",
                        default=3,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("Setting the random seed...")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = os.path.join(args.output_dir, "length_512.py")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    log_path = os.path.join(output_dir, "log")

    print("Reading data...")
    df_data = pd.read_csv(args.train_data_path, sep="\t")
    data = df_data['data'].tolist()
    label_set = sorted(list(set(df_data['label'].values)))
    labels = encode_labels(df_data['label'].tolist(), label_set)
    num_labels = len(set(labels))
    acc = []
    f1 = []
    cro_acc = []
    cro_f1 = []

    print("Preparing the croatian test data...")
    cro_test_data = []
    cro_test_labels = []
    with open(args.cro_test_data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for i, line in enumerate(reader):
            if i == 0:
                continue
            cro_test_labels.append(line[0])
            cro_data = line[1] + ". " + line[2]
            cro_test_data.append(cro_data)
    cro_test_labels = encode_labels(cro_test_labels, label_set)


    for i in range(10):
        print("Training model on the split number " + str(i) + "...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)

        output_subdir = os.path.join(output_dir, str(i))
        print(output_subdir)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)

        test_data = data[(floor(len(data) * i * 0.1)):(floor(len(data) * (i + 1) * 0.1))]
        test_labels = labels[floor((len(labels) * i * 0.1)):floor((len(labels) * (i + 1) * 0.1))]
        train_data = data[:floor((len(data) * i * 0.1))] + data[floor((len(data) * (i + 1) * 0.1)):]
        train_labels = labels[:floor((len(labels) * i * 0.1))] + labels[floor((len(labels) * (i + 1) * 0.1)):]
        train_data, eval_data, train_labels, eval_labels = train_test_split(train_data, train_labels,
                                                                    test_size=args.eval_split, random_state=42)
        print("Train label:")
        print(train_labels[0])
        print("Train data:")
        print(train_data[0])
        train_dataloader = cut_at_length(train_data, train_labels, tokenizer, args.max_len, args.batch_size)
        eval_dataloader = cut_at_length(eval_data, eval_labels, tokenizer, args.max_len, args.batch_size)
        test_dataloader = cut_at_length(test_data, test_labels, tokenizer, args.max_len, args.batch_size)
        cro_test_dataloader = cut_at_length(cro_test_data, cro_test_labels, tokenizer, args.max_len, args.batch_size)
        _, __ = bert_train(model, device, train_dataloader, eval_dataloader, output_subdir, args.num_epochs,
                           args.warmup_proportion, args.weight_decay, args.learning_rate, args.adam_epsilon,
                           save_best=True)

        print("Testing the trained model on the current test split...")
        metrics = bert_evaluate(model, test_dataloader, device)
        with open(log_path, 'a') as f:
            f.write("Results for split nr. " + str(i) + " on current slo test:\n")
            f.write("Acc: " + str(metrics['accuracy']) + "\n")
            f.write("F1: " + str(metrics['f1']) + "\n")
            f.write("\n")
        acc.append(metrics['accuracy'])
        f1.append(metrics['f1'])

        print("Testing the trained model on the croatian test set...")
        cro_metrics = bert_evaluate(model, cro_test_dataloader, device)
        with open(log_path, 'a') as f:
            f.write("Results for split nr. " + str(i) + " on cro test set:\n")
            f.write("Acc: " + str(cro_metrics['accuracy']) + "\n")
            f.write("F1: " + str(cro_metrics['f1']) + "\n")
            f.write("\n")
        cro_acc.append(cro_metrics['accuracy'])
        cro_f1.append(cro_metrics['f1'])


    avg_acc = np.mean(acc)
    avg_f1 = np.mean(f1)
    avg_cro_acc = np.mean(cro_acc)
    avg_cro_f1 = np.mean(cro_f1)
    print("Avg. acc: " + str(avg_acc))
    print("Avg. F1: " + str(avg_f1))
    print("Avg. acc on cro test set: " + str(avg_cro_acc))
    print("Avg. f1 on cro test set: " + str(avg_cro_f1))
    print("Writing the results...")
    with open(log_path, 'a') as f:
        f.write("Avg. acc: " + str(avg_acc) + "\n")
        f.write("Avg. F1: " + str(avg_f1) + "\n")
        f.write("Avg. acc on cro test set: " + str(avg_cro_acc) + "\n")
        f.write("Avg. f1 on cro test set: " + str(avg_cro_f1) + "\n")
        f.write("\n")
    print("Done.")


if __name__ == "__main__":
    length_512_crossvalidation()
