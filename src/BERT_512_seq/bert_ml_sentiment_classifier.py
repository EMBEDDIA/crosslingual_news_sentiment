import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from metrics import get_metrics

from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import os
import numpy as np


def bert_train(model, device, train_dataloader, eval_dataloader, output_dir, num_epochs, warmup_proportion, weight_decay,
               learning_rate, adam_epsilon, save_best=False):
    """Training loop for bert fine-tuning. Save best works with F1 only currently."""

    t_total = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_proportion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    train_iterator = trange(int(num_epochs), desc="Epoch")
    model.to(device)
    tr_loss_track = []
    eval_metric_track = []
    output_filename = os.path.join(output_dir, 'pytorch_model.bin')
    f1 = float('-inf')

    for _ in train_iterator:
        model.train()
        model.zero_grad()
        tr_loss = 0
        nr_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            tr_loss = 0
            input_ids, input_mask, labels = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=input_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nr_batches += 1
            model.zero_grad()

        print("Evaluating the model on the evaluation split...")
        metrics = bert_evaluate(model, eval_dataloader, device)
        eval_metric_track.append(metrics)
        if save_best:
            if f1 < metrics['f1']:
                model.save_pretrained(output_dir)
                torch.save(model.state_dict(), output_filename)
                print("The new value of f1 score of " + str(metrics['f1']) + " is higher then the old value of " +
                      str(f1) + ".")
                print("Saving the new model...")
                f1 = metrics['f1']
            else:
                print("The new value of f1 score of " + str(metrics['f1']) + " is not higher then the old value of " +
                      str(f1) + ".")

        tr_loss = tr_loss / nr_batches
        tr_loss_track.append(tr_loss)

    if not save_best:
        model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_filename)

    return tr_loss_track, eval_metric_track


def bert_evaluate(model, eval_dataloader, device):
    """Evaluation of trained checkpoint."""
    model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    data_iterator = tqdm(eval_dataloader, desc="Iteration")
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        #loss is only output when labels are provided as input to the model
        logits = outputs[0]
        print(type(logits))
        logits = logits.to('cpu').numpy()
        label_ids = labels.to('cpu').numpy()

        for label, logit in zip(label_ids, logits):
            true_labels.append(label)
            predictions.append(np.argmax(logit))

    #print(predictions)
    #print(true_labels)
    metrics = get_metrics(true_labels, predictions)
    return metrics


def bert_predict(model, test_dataloader, device):
    """Testing of trained checkpoint.
    CHANGE SO IT DOES NOT TAKE DATALOADER WITH LABELS BECAUSE WE DON'T NEED THEM"""
    model.to(device)
    model.eval()
    predictions = []
    #true_labels = []
    data_iterator = tqdm(test_dataloader, desc="Iteration")
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        #loss is only output when labels are provided as input to the model
        logits = outputs[:2]
        print(type(logits))
        logits = logits.to('cpu').numpy()
        #label_ids = labels.to('cpu').numpy()

        for logit in logits:
            #true_labels.append(label)
            predictions.append(np.argmax(logit))

    #print(predictions)
    #print(true_labels)
    #metrics = get_metrics(true_labels, predictions)
    return predictions


def bert_extract_CLS_embedding(model, dataloader, device):
    """For each datapoint extracts embeddings for CLS token from the last layer and returns a list of CLS embeddings
    NOTE: input model should have the option 'output_hidden_states' set to True when initialized
    """
    model.to(device)
    model.eval()
    cls_embedding = []
    for step, batch in enumerate(dataloader):
        input_ids, input_mask = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        all_hidden_states = outputs[-1]
        last_layer = all_hidden_states[-1]
        last_layer = last_layer.to('cpu').numpy()
        print("Shape:")
        print(last_layer.shape)
        cls_embedding.append(last_layer[0])
        print(cls_embedding.shape)

    return cls_embedding

