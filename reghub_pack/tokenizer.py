
from transformers import BertTokenizer
from transformers import RobertaTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")





tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer_RoBERTa = RobertaTokenizer.from_pretrained('roberta-base')

