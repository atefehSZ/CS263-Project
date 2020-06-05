import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel
from transformers import BertTokenizer
from torchtext import data

def tokenize_and_cut(sentence):
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

def package_data(X: pd.DataFrame, tokenizer: BertTokenizer):
    # special tokens
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    
    TEXT = data.Field(batch_first=True,
              use_vocab=False,
              tokenize=tokenize_and_cut,
              preprocessing=tokenizer.convert_tokens_to_ids,
              init_token=init_token_idx,
              eos_token=eos_token_idx,
              pad_token=pad_token_idx,
              unk_token=unk_token_idx)
    LABEL = data.RawField(is_target=True)
    # LABEL = data.LabelField(dtype=torch.float)

    newsdata = NewsDataset(X, TEXT, LABEL)
    return newsdata

class NewsDataset(data.Dataset):
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, df, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for i, row in df.iterrows():
            examples.append(
                data.Example.fromlist([row['sentences'], row['scores']], fields))

        super(NewsDataset, self).__init__(examples, fields, **kwargs)

class BertSentimentAnalyzer(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int,
                 n_layers: int, bidirectional: bool, dropout: int):
        super().__init__()
        
        # Use pre-trained BERT to get embeddings
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained(BERT_N)
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        # freeze bert params
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bias = True,
                            batch_first = True,
                            dropout = 0 if n_layers < 2 else dropout,
                            bidirectional = bidirectional)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, text):
        # text = (batch size, sentence length)
        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = (batch size, sent len, emb dim)
        _, (hidden, _) = self.lstm(embedded)

        # hidden = (n layers * n directions, batch size, emb dim)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        # hidden = (batch size, hid dim)
        # tanh because our outputs are between -1 and 1 from nltk
        output = torch.tanh(self.out(hidden))

        # output = (batch size, out dim)
        return output.squeeze(1)

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
        ground_truth = torch.tensor(batch.label, device=device)
        
        optimizer.zero_grad()
        score = model(batch.text)
#         print("score")
#         print(score)
#         print("true")
#         print(ground_truth)
        loss = lossf(score, ground_truth)
        print(loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    epoch_i_loss = train_loss / len(train_loader)
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
            ground_truth = torch.tensor(batch.label, device=device)
            score = model(batch.text)
            test_loss += lossf(score, ground_truth).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> {dataset} ' + 'set loss: {:.4f}'.format(test_loss))
    return test_loss

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience.

    Based off the code by Github user Bjarten, in the following repo:
        https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0
    ):
        """
        Args:
            patience: long to wait after last time validation loss improved.
            verbose: If True prints message for each val loss improvement.
            delta: Min change in monitored quantity to qualify as improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0