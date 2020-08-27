import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, LogSoftmax
from tqdm import tqdm, trange

import numpy as np

from metrics import get_metrics

class TwoLayerFCNet(torch.nn.Module):
    """A two layer fully-connected classifier with relu activation for hidden
    layer and softmax activation for output layer.
    """
    def __init__(self):
        super(TwoLayerFCNet, self).__init__()
        self.hidden = torch.nn.Linear(768, 250)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(250, 3)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        h1 = self.hidden(x)
        h1 = self.relu(h1)
        out = self.output(h1)
        activation = self.softmax(out)
        return activation


class CNN_net(torch.nn.Module):
    """A two layer fully-connected classifier with relu activation for hidden
    layer and softmax activation for output layer.
    """
    def __init__(self):
        super(CNN_net, self).__init__()
        self.convolution = torch.nn.Conv1d(768, 128, 2, stride=2, padding=0)
        #self.batch_norm = torch.nn.BatchNorm1d(50)
        self.relu = torch.nn.ReLU()
        #self.max_pool = torch.nn.MaxPool1d(kernel_size=50),
        self.output = torch.nn.Linear(128, 3)
        #self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        cnn = self.convolution(x)
        h1 = self.relu(cnn)
        #print(h1.shape)
        max = torch.max(h1, dim=2)
        #print(type(max[0]))
        #print(max[0].shape)
        #max = torch.tensor(max)
        #print(h1.size())
        out = self.output(max[0])
        #print(out.size())
        activation = self.relu(out)
        #print(activation.size())
        activation = activation.squeeze()
        #out.squeeze()
        return activation
        #return out

class WeightedSum(torch.nn.Module):
    """Optimizing the weighted sum of the input embedding sequences
    """
    def __init__(self):
        super(WeightedSum, self).__init__()
        self.weights = torch.nn.Parameter(torch.rand(1, 6), requires_grad=True)
        self.hidden = torch.nn.Linear(768, 250)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(250, 3)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        print(x.size())
        print(self.weights.size())
        sum = (self.weights * x).sum()
        h1 = self.hidden(sum)
        h1 = self.relu(h1)
        out = self.output(h1)
        activation = self.softmax(out)
        return activation


class ClassificationHead:
    """A trainer class for simple classification models. Needed to instantiate with a concrete
    model architecture. Able to save or load trained models provided their architecture is the same as
    it was instantiated in this class.
    """
    def __init__(self, model, device, num_epochs, learning_rate, weight_decay, adam_epsilon):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, train_dataloader, eval_dataloader, output_model_path, save_best=True):
        no_decay = ['bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        criterion = CrossEntropyLoss()
        epoch_iterator = trange(self.num_epochs, desc="Epoch")

        tr_loss_track = []
        eval_metric_track = []
        f1 = float('-inf')

        self.model.to(self.device)
        for _ in epoch_iterator:
            self.model.train()
            self.model.zero_grad()
            tr_loss = 0
            nr_batches = 0
            batch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(batch_iterator):
                tr_loss = 0
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                #this is probably redundant
                optimizer.zero_grad()

                outputs = self.model(input_ids)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                nr_batches += 1
                self.model.zero_grad()

            print("Evaluating the model on the evaluation split...")
            metrics = self.evaluate(eval_dataloader)
            eval_metric_track.append(metrics)
            if save_best:
                if f1 < metrics['f1']:
                    self.save_weights(output_model_path)
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
            self.save_weights(output_model_path)

        return tr_loss_track, eval_metric_track

    def evaluate(self, eval_dataloader):
        """Evaluation of trained checkpoint."""

        self.model.to(self.device)
        self.model.eval()
        predictions = []
        true_labels = []
        data_iterator = tqdm(eval_dataloader, desc="Iteration")
        for step, batch in enumerate(data_iterator):
            input_ids, labels = batch
            input_ids = input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
            outputs = outputs.to('cpu').numpy()
            print(type(outputs))
            print("Shape:")
            print(outputs.shape)
            label_ids = labels.to('cpu').numpy()

            for label, output in zip(label_ids, outputs):
                true_labels.append(label)
                predictions.append(np.argmax(output))

        # print(predictions)
        # print(true_labels)
        metrics = get_metrics(true_labels, predictions)
        return metrics

    def predict(self):
        pass
