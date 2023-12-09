
# ! pip install transformers
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer

from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer

import torch

import pandas as pd
import numpy as np




import warnings
warnings.filterwarnings("ignore")





# import BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class Dataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.labels=df['target']
        self.text=[tokenizer(text,padding='max_length',truncation=True,return_tensors="pt") for text in df['news_content']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.text[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y