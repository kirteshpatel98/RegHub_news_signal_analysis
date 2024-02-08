# ! pip install transformers
import torch
import numpy as np
import matplotlib.pyplot as plt
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW

import torchsummary as summary
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
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
from ..tokenizer import tokenizer_LLaMA



# BERT classifier architecture, with 7 output classes
base_model_path="./huggingface/llama7B"
model_name = "meta-llama/Llama-2-7b-chat-hf"


model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)



class LLaMA2(nn.Module):

    def __init__(self):

        super(LLaMA2, self).__init__()
        base_model_path="./huggingface/llama7B"
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.llama = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)

    def forward(self, input_ids, attention_mask):

        output = self.llama(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)
        
        return output
'''

class LLaMA2(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, model_name, *model_args, **kwargs):
        # This method will handle the model initialization
        return super(LLaMA2, cls).from_pretrained(model_name, *model_args, **kwargs, device_map='auto',torch_dtype=torch.float16)

    def forward(self, input_ids, attention_mask=None):
        return super(LLaMA2, self).forward(input_ids=input_ids, attention_mask=attention_mask)

'''


class LLaMA_RegHub(LLaMA2):
    def __init__(self): 
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        super(LLaMA_RegHub, self).__init__()
        # super(LLaMA_RegHub, self).from_pretrained(model_name)
        
    def load_model(self,model_name='LLaMA_generative.pth',torch=torch,from_aws=False,bucket="fs-reghub-news-analysis"):
        if from_aws:
            with open("../aws_credentials.json", 'r') as file:
                aws_creds_json = json.load(file)
        
            aws = awsOps(aws_creds_json)
            aws.download_file(bucket=bucket, file=model_name, output=model_name)
            
        checkpoint = torch.load(model_name)
        self.load_state_dict(checkpoint['model_state_dict'])

    def display_model_layers(self):
        # display LLaMA layers
        n=0
        for x in self.state_dict():
            n=n+1
            print(x)
        print(n)
    
    def freeze_layers(self,lay=250):
        # freeze first 8 layers 
        self.lay=lay
        n=0
        for param in self.parameters():
            n=n+1
            param.requires_grad = False
            if n==(self.lay):
                break
   
    def hyper_parameters(self,epochs=100,LR=0.00001,criterion=CrossEntropyLoss(),optimizer=Adam):
        # hyperparameters
        self.epochs=epochs
        self.LR=LR
        # self.criterion=criterion.to(self.device)
        self.optimizer=optimizer(self.parameters(), lr= self.LR)
                
              
    def pre_load(self, train_data=None, val_data=None, batch_size=32,Dataset=Dataset_CLM,DataLoader=DataLoader,tokenizer=tokenizer_LLaMA):
        
        self.batch_size=batch_size
        self.train_data=train_data
        self.val_data=val_data
        # initialize parent class Dataset
        self.train=Dataset(self.train_data,tokenizer=tokenizer)
        self.val=Dataset(self.val_data,tokenizer=tokenizer)
        
        
        self.train_dataloader = DataLoader(self.train, batch_size=self.batch_size)
        self.val_dataloader = DataLoader(self.val, batch_size=self.batch_size)
        
    
    def model_training(self,train_data=None, val_data=None,freeze=True,tqdm=tqdm):
        
        if freeze==True:
            self.freeze_layers(lay=68)
            
        self.hyper_parameters()
        self.pre_load(train_data=train_data, val_data=val_data)
        
        self.plot_val_acc=[]
        self.plot_train_acc=[]
        self.plot_epoch=[]
        self.plot_train_loss=[]
        self.plot_val_loss=[]
        
        for self.epoch_num in range(self.epochs): 
            
            self.total_loss_train = 0

            self.true_acc=0
            n=0
            
            with tqdm(self,total=len(self.train_dataloader), desc=f'Epoch {self.epoch_num + 1}/{self.epochs}', unit='item',position=0,leave=True) as p_bar:
                for batch in self.train_dataloader:
                    input_ids = batch["input_ids"].squeeze(1).to("cuda:0")
                    attention_mask = batch["attention_mask"].squeeze(1).to("cuda:0")

                    # Prepare labels (shift input_ids to the left)
                    # labels = input_ids[:, 1:].contiguous()
                    # input_ids = input_ids[:, :-1].contiguous()
                    # attention_mask = attention_mask[:, :-1].contiguous()
                    labels=input_ids
                    # print(labels[0])
                    print(input_ids[0])
                    print(labels[0])
                    print(attention_mask[0])
                    
                    '''
                    train_label = train_input['input_ids'].to("cuda:0") # to cuda GPU
                    input_id = train_input['input_ids'].to("cuda:0")
                    mask = train_input['attention_mask'].to("cuda:0")
                    
                    labels = input_id[:, 1:].contiguous()
                    input_id = input_id[:, :-1].contiguous()
                    mask = attention_mask[:, :-1].contiguous()
                    input_id = train_input['input_ids'].to("cuda:0")
                    '''

                    l1_loss=0
                    
                    
                    # for L1 regularization
                    a=0
                    reg_loss = 0
                    for param in self.parameters():
                        a=a+1
                        if a >=201-68:
                            reg_loss += torch.norm(param, 1)
                            
                    # model output
                    output = self(input_ids,labels)
                    
                    
                    # loss value
                    batch_loss = self.criterion(output, train_label) + l1_loss
                    self.total_loss_train += batch_loss.item() 
                    
                    # train accuracy 
                    output=output.squeeze()
                    train_label=train_label.squeeze()
                    # print(output)
                    # print(train_label)
                    # print(torch.gather(train_label, 1, torch.argmax(output,dim=1).view(-1, 1)))
                    acc = torch.gather(train_label, 1, torch.argmax(output,dim=1).view(-1, 1)).sum().item()
                    self.total_acc_train += acc
                    
                    # true accuracy
                    self.true_acc_batch=0
                    for (x,y) in zip(torch.round(F.sigmoid(output)*10)/10,train_label): # ceil or round
                        x=torch.tensor(x)
                        y=torch.tensor(y)
                        # print(x)
                        # print(y)
                        x[x<1]=0
                        self.true_acc=self.true_acc+torch.sum(x==y).item()
                        self.true_acc_batch=self.true_acc_batch+torch.sum(x==y).item()
                        
                        
                    # backpropogation
                    self.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    p_bar.set_postfix(loss=batch_loss.item() / len(train_input['input_ids']), acc=acc / len(train_input['input_ids']), true_acc=self.true_acc_batch / (len(train_input['input_ids'])*9))
                    p_bar.update()
                
            self.model_validation()        
            try:
                if self.plot_val_loss[-1]>self.plot_val_loss[-2] and self.plot_val_loss[-2]>self.plot_val_loss[-3]:
                    break
            except:
                pass
            
    def model_validation(self):
        # for validation accuracy
        
        self.total_acc_val = 0
        self.total_loss_val = 0
        self.true_val_acc=0
        with torch.no_grad():

            for val_input, val_label in self.val_dataloader:

                val_label = val_label.to(self.device)
                mask = val_input['attention_mask'].to(self.device)
                input_id = val_input['input_ids'].squeeze(1).to(self.device)

                # validation output
                output = self(input_id, mask)

                # validation loss value
                batch_loss = self.criterion(output, val_label)
                self.total_loss_val += batch_loss.item()

                # validation accuracy
                output=output.squeeze()
                val_label=val_label.squeeze()
                acc = torch.gather(val_label, 1, torch.argmax(output,dim=1).view(-1, 1)).sum().item()
                self.total_acc_val += acc

                # true accuracy
                for (x,y) in zip(torch.round(F.sigmoid(output)*10)/10,val_label): # ceil or round
                    x=torch.tensor(x)
                    y=torch.tensor(y)
                    x[x<1]=0
                    self.true_val_acc=self.true_val_acc+torch.sum(x==y).item()
                    
        self.print_metrics()        
                    
                    
    def print_metrics(self):
        print(
            f'Epochs: {self.epoch_num + 1} | Train Loss: {self.total_loss_train / len(self.train_data): .3f} \
            | Train Accuracy: {self.total_acc_train / len(self.train_data): .3f} \
            | Val Loss: {self.total_loss_val / len(self.val_data): .3f} \
            | Val Accuracy: {self.total_acc_val / len(self.val_data): .3f} \
            | True Train Accuracy: {self.true_acc / (len(self.train_data)*9): .3f} \
            | True Val Accuracy: {self.true_val_acc / (len(self.val_data)*9): .3f}',end='\r')
        self.plot_train_acc.append(self.total_acc_train / len(self.train_data))
        self.plot_val_acc.append(self.total_acc_val / len(self.val_data))
        self.plot_epoch.append(self.epoch_num + 1)
        self.plot_train_loss.append(self.total_loss_train / (len(self.train_data)*9))
        self.plot_val_loss.append(self.total_loss_val / (len(self.val_data)*9))
        
        self.model_checkpoint_save()
        
        
    
    def metrics_graph(self,plt=plt):
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        fig.suptitle('Accuracy and Loss')



        ax[0].plot(self.plot_epoch,self.plot_val_acc, label='Validation Accuracy',color='orange')
        ax[0].plot(self.plot_epoch,self.plot_train_acc, label='Training Accuracy',color='red')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(self.plot_epoch,self.plot_train_loss, label=' Training Loss',color='blue')
        ax[1].plot(self.plot_epoch,self.plot_val_loss, label=' Validation Loss',color='yellow')
        ax[1].legend()
        ax[1].grid(True)

        plt.show()
        
        
        
    def model_checkpoint_save(self):
        # model checkpoint
        checkpoint = {'epoch': self.epoch_num+1, 'model_state_dict': self.state_dict() }

        torch.save(checkpoint, f'BERT_checkpoint_epoch{self.epoch_num+1}.pth')

        
        
        
    def save_best_model(self,save_aws=True,name='BERT_classifier.pth'):
        # retrieve best checkpoint model
        self.name=name
        checkpoint = torch.load(f'BERT_checkpoint_epoch{self.epoch_num+1-2}.pth')
        
        # self.load_state_dict(checkpoint['model_state_dict'])


        # save best model
        torch.save(checkpoint, self.name)
        # torch.save(self, self.name)
        
        # delete remaining model checkpoints
        for f in glob.glob("BERT_checkpoint*.pth"):
            os.remove(f)
        
        if save_aws:
            self.save_to_aws()
    
    def save_to_aws(self,bucket="fs-reghub-news-analysis",awsOps=awsOps):
        with open("../aws_credentials.json", 'r') as file:
            aws_creds_json = json.load(file)
        
        aws = awsOps(aws_creds_json)
        aws.upload_file(bucket=bucket, path=self.name, name=self.name)
        
    
    
    def classifier(self,F=F,torch=torch,input_text="Commerzbank faced bankrupcy today",table_format=False,df_test=None,Dataset=Dataset,pd=pd,custom_model=None,tokenizer=tokenizer_LLaMA):
        
        if custom_model != None:
            self.load_model(model_name=custom_model,torch=torch)
            
        if table_format==True:
            df_test['target']=1
            batch_size=len(df_test)
        else:
            df_test=pd.DataFrame({'news_content':[input_text],'target':[1]})
            batch_size=1
               
        test = Dataset(df_test,tokenizer=tokenizer)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
        
        
        with torch.no_grad():
            for test_input,_ in test_dataloader:
                mask = test_input['attention_mask'].to(self.device)
                input_id = test_input['input_ids'].squeeze(1).to(self.device)
                output = self(input_id, mask)
                output=F.softmax(output).squeeze()*100
                if table_format==True:
                    df_test['legal']=output[0].item()
                    df_test['sanctions']=output[1].item()
                    df_test['papers']=output[2].item()
                    df_test['reports']=output[3].item()
                    df_test['statements']=output[4].item()
                    df_test['guidelines']=output[5].item()
                    df_test['press']=output[6].item()
                    df_test['personnel']=output[7].item()
                    df_test['market']=output[8].item()
                    return df_test
                
                else:
                    print('legal',output[0].item(),'%')
                    print('sanctions',output[1].item(),'%')
                    print('papers',output[2].item(),'%')
                    print('reports',output[3].item(),'%')
                    print('statements',output[4].item(),'%')
                    print('guidelines',output[5].item(),'%')
                    print('press',output[6].item(),'%')
                    print('personnel',output[7].item(),'%')
                    print('market',output[8].item(),'%')