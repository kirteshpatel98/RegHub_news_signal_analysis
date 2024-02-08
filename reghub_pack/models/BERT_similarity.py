# ! pip install transformers
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW

import torchsummary as summary
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import BertModel


from sklearn.model_selection import train_test_split
import torch

import pandas as pd
import numpy as np
import os
import base64
import gc
import glob, os

from torch.optim import Adam
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import json
from ..Dataset import *
from ..general_functions import *
from ..tokenizer import tokenizer_BERT



# BERT classifier architecture, with 7 output classes
class BertMLM(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertMLM, self).__init__()

        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased")


    def forward(self, input_id, mask):

        output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)

        return output
    

class BertSimilarity(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertSimilarity, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')


    def forward(self, input_id, mask):

        output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)

        return output




class BERT_RegHub_Similarity(BertSimilarity,BertMLM):
    def __init__(self,):            
        BertSimilarity.__init__(self)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device=device
        self.to(self.device)
    def load_mlm(self):
        BertMLM.__init__(self)
        use_cuda = torch.cuda.is_available()
        self.bert.to(self.device)
      
    def load_model(self,model_name='BERT_similarity.pth',torch=torch,from_aws=False,bucket="fs-reghub-news-analysis"):
        if from_aws:
            with open("../aws_credentials.json", 'r') as file:
                aws_creds_json = json.load(file)
        
            aws = awsOps(aws_creds_json)
            aws.download_file(bucket=bucket, file=model_name, output=model_name)
            
        checkpoint = torch.load(model_name)
        self.load_state_dict(checkpoint['model_state_dict'])

    def display_model_layers(self):
        # display BERT layers
        n=0
        for x in self.state_dict():
            n=n+1
            print(x)
        print(n)
    
    def freeze_layers(self,lay=68):
        # freeze first 8 layers 
        self.bert.lay=lay
        n=0
        for param in self.bert.parameters():
            n=n+1
            param.requires_grad = False
            if n==(201-self.bert.lay):
                break
   
    def hyper_parameters(self,epochs=10,LR=0.00001,criterion=CrossEntropyLoss(),optimizer=Adam):
        # hyperparameters
        self.bert.epochs=epochs
        self.bert.LR=LR
        self.bert.criterion=criterion.to(self.bert.device)
        self.bert.optimizer=optimizer(self.bert.parameters(), lr= self.bert.LR)
                
              
    def pre_load(self, train_data=None, val_data=None, batch_size=15,Dataset=Dataset_MLM,DataLoader=DataLoader,tokenizer=tokenizer_BERT):
        
        self.bert.batch_size=batch_size
        self.bert.train_data=train_data
        self.bert.val_data=val_data
        self.tokenizer = tokenizer
        
        # initialize parent class Dataset
        self.bert.train_d=Dataset(df = self.bert.train_data,tokenizer=self.tokenizer)
        self.bert.val_d=Dataset(df = self.bert.val_data,tokenizer=self.tokenizer)
        
        
        self.bert.train_dataloader = DataLoader(self.bert.train_d, batch_size=self.bert.batch_size)
        self.bert.val_dataloader = DataLoader(self.bert.val_d, batch_size=self.bert.batch_size)
        
    
    def model_training(self,train_data=None, val_data=None,freeze=True,tqdm=tqdm):
        
        if freeze==True:
            self.freeze_layers(lay=68)
            
        self.hyper_parameters()
        self.pre_load(train_data=train_data, val_data=val_data)
        
        self.bert.plot_train_loss=[]
        self.bert.plot_val_loss=[]
        
        for self.bert.epoch_num in range(self.bert.epochs): 
            
            self.bert.total_loss_train = 0

            n=0
            
            with tqdm(self,total=len(self.bert.train_dataloader), desc=f'Epoch {self.bert.epoch_num + 1}/{self.bert.epochs}', unit='item',position=0,leave=True) as p_bar:
                for train_input, train_label in self.bert.train_dataloader:
                    self.bert.optimizer.zero_grad()
                    
                    train_label = train_label['input_ids'].to(self.device) # to cuda GPU
                    mask = train_input['attention_mask'].to(self.device) # attention mask
                    input_id = train_input['input_ids'].squeeze(1).to(self.device)

                    # model output
                    outputs = self.bert(input_ids = input_id, attention_mask=mask,labels=train_label)
                    
                    batch_loss = outputs.loss
                    batch_loss.backward()
                    self.bert.optimizer.step()
                    
                    # loss value
                    self.bert.total_loss_train += batch_loss.item()  
               
                    p_bar.set_postfix(loss=batch_loss.item() / len(train_input['input_ids']))
                    p_bar.update()
                
            self.model_validation()        
            try:
                if self.bert.plot_val_loss[-1]>self.bert.plot_val_loss[-2] and self.bert.plot_val_loss[-2]>self.bert.plot_val_loss[-3]:
                    break
            except:
                pass
            
        self.delete_MLM()
        self.weight_transfer()
        self.save_model(save_aws=False)
            
    def model_validation(self):
        # for validation accuracy
        
        self.bert.total_loss_val = 0
        with torch.no_grad():

            for val_input, val_label in self.bert.val_dataloader:

                val_label = val_label['input_ids'].to(self.device)
                mask = val_input['attention_mask'].to(self.device)
                input_id = val_input['input_ids'].squeeze(1).to(self.device)

                outputs = self.bert(input_ids = input_id, attention_mask=mask,labels=val_label)

                batch_loss = outputs.loss                                        
                # validation loss value
                self.bert.total_loss_val += batch_loss.item()

               
                    
        self.print_metrics() 
        
                    
                    
    def print_metrics(self):
        print(
            f'Epochs: {self.bert.epoch_num + 1} | Train Loss: {self.bert.total_loss_train / len(self.bert.train_data): .3f} \
            | Val Loss: {self.bert.total_loss_val / len(self.bert.val_data): .3f}',end='\r')         
        self.model_checkpoint_save()
        
        

        
        
    def weight_transfer(self,):
        #retrieve the layers
        n1=0
        del_layers=[]
        for x in self.bert.state_dict():
            n1=n1+1
            if n1>197:
                del_layers.append(str(x))

        # delete the lyers
        state_dict = self.bert.state_dict()
        for key in del_layers:
            if key in state_dict:
                del state_dict[key]

        # transfer weights
        for x, y in zip(state_dict,self.state_dict()):
            for n in range(len(self.state_dict()[str(y)])):
                self.state_dict()[str(y)][n] = state_dict[str(x)][n]
        
    def model_checkpoint_save(self):
        # model checkpoint
        checkpoint = {'epoch': self.bert.epoch_num+1, 'model_state_dict': self.bert.state_dict() }

        torch.save(checkpoint, f'BERT_MLM_checkpoint_epoch{self.bert.epoch_num+1}.pth')
             
        
    def delete_MLM(self,name='BERT_similarity.pth'):
        # retrieve best checkpoint model
        self.name=name
        checkpoint = torch.load(f'BERT_MLM_checkpoint_epoch{self.bert.epoch_num+1-2}.pth')
        
        self.bert.load_state_dict(checkpoint['model_state_dict'])
        
        # delete remaining model checkpoints
        for f in glob.glob("BERT_MLM_checkpoint_epoch*.pth"):
            os.remove(f)
            
            
    def save_model(self,save_aws=True,name='BERT_similarity1.pth'):
        # retrieve best checkpoint model
        self.name=name
        final_checkpoint={'model_state_dict': self.state_dict() }
        # save best model
        torch.save(final_checkpoint, self.name)
        
        if save_aws:
            self.save_to_aws()
        
    
    def save_to_aws(self,bucket="fs-reghub-news-analysis",awsOps=awsOps):
        with open("../aws_credentials.json", 'r') as file:
            aws_creds_json = json.load(file)
        
        aws = awsOps(aws_creds_json)
        aws.upload_file(bucket=bucket, path=self.name, name=self.name)
        
    def cosine_similarity(self,tensor_a, tensor_b):
        # normalize 
        norm_a = torch.norm(tensor_a)
        normalized_a = tensor_a / norm_a

        # normalize 
        norm_b = torch.norm(tensor_b)
        normalized_b = tensor_b / norm_b

        # dot product
        cos_similarity = torch.dot(normalized_a.flatten(), normalized_b.flatten())

        return cos_similarity
        
    def similarity_output(self,text1,text2):
        input1 = self.tokenizer(text1)
        input2 = self.tokenizer(text2)
        
        output1 = self(**input1)
        output2 = self(**input2)

        output1 = output1[0].squeeze() 
        output2 = output2[0].squeeze()
        
        print(self.cosine_similarity(output1, output2))