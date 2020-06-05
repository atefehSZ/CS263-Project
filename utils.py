import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torchtext import data

def sst_threshold(val):
    if val > 0.6:
        return 2 # pos
    elif val < 0.4:
        return 1 # neu
    else:
        return 0 # neg

def to_sentiment(classifier, sentence):
    """neg:0, neu:1, pos:2"""
    scores = classifier.polarity_scores(sentence)
    return np.argmax([scores['neg'], scores['neu'], scores['pos']])
#     return [scores['neg'], scores['neu'], scores['pos']]

def get_predictions(model, data_loader, device):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d['sentence']
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['label'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids=None)
            _, preds = torch.max(outputs[0], dim=1)
#             _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs[0], dim=1)
#             probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');

def package_data(
    X: pd.DataFrame,
    tokenizer: BertTokenizer,
    max_len: int,
    batch_size: int,
) -> DataLoader:
    """Pytorch modules require DataLoaders for train/val/test"""
    dataset = NewsDataset(X['sentences'].to_numpy(),
                          X['scores'].to_numpy(),
                          tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size,
                        num_workers=4)

class NewsDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        tokenizer: BertTokenizer,
        max_len: int,
    ):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sentence = self.data[idx]
        label = self.labels[idx]

        encoded = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
#             return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'sentence': sentence,
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertSentimentAnalyzer(nn.Module):
    def __init__(self, output_dim: int, dropout: float):
        super(BertSentimentAnalyzer, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-cased')
        hidden_dim = self.bert.config.hidden_size
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
        # freeze bert layers
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, input_ids, mask):
        _, pooled_output = self.bert(
#         context_rep , _ = self.bert(
            input_ids=input_ids,
            attention_mask=mask
        )
#         cls_rep = context_rep[:,0]
#         output = self.dropout(cls_rep)
#         output  = self.out(cls_rep)
        output = self.dropout(pooled_output)
        output = self.out(output)
#         return torch.tanh(self.out(output)).squeeze(1)
#         return self.softmax(output)
        # CrossEntropyLoss takes the logits
        return output

def train(
    model: BertSentimentAnalyzer,
    epoch: int,
    optimizer: optim.Optimizer,
    train_loader: data.BucketIterator,
    batch_log_interval: int,
    lossf: nn.Module,
    device: str = 'cpu',
) -> float:
    """Runs train for n epochs and returns epoch losses."""
    model.train()
    train_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        ground_truth = batch['label'].to(device)
        
        optimizer.zero_grad()
        loss, logits = model(input_ids,
                             token_type_ids=None,
                             attention_mask=mask,
                             labels=ground_truth)
#         scores = model(input_ids, mask)
#         loss = lossf(scores, ground_truth)
        loss.backward()
        train_loss += loss.item()
        # avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    epoch_i_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, epoch_i_loss))
    return epoch_i_loss

def evaluate(
    model: BertSentimentAnalyzer,
    epoch: int,
    test_loader: data.BucketIterator,
    lossf: nn.Module,
    device: str = 'cpu',
    is_val: bool = False,  # is validation
) -> float:
    """Runs test for n epochs and returns a list of epoch losses."""
    dataset = 'Val' if is_val else 'Test'
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ground_truth = batch['label'].to(device)
            loss, logits = model(input_ids,
                             token_type_ids=None,
                             attention_mask=mask,
                             labels=ground_truth)
#             scores = model(input_ids, mask)
#             scores = model(input_ids, mask).squeeze(-1)
#             test_loss += lossf(scores, ground_truth).item()
#             test_loss += lossf(scores, ground_truth.float()).item()
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print(f'====> {dataset} ' + 'set loss: {:.4f}'.format(test_loss))
    return test_loss
