
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

model_name = "meta-llama/Llama-2-7b-chat-hf"

base_model_path="./huggingface/llama7B"



tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer_RoBERTa = RobertaTokenizer.from_pretrained('roberta-base')

tokenizer_LLaMA = AutoTokenizer.from_pretrained(model_name)
if tokenizer_LLaMA.pad_token is None:
    tokenizer_LLaMA.pad_token = tokenizer_LLaMA.eos_token
    
tokenizer_LLaMA.padding_side = 'right'  # You can use 'right' or 'left'
