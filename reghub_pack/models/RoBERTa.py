# ! pip install transformers
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchsummary as summary
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel

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
from ..tokenizer import tokenizer_RoBERTa


# BERT classifier architecture, with 7 output classes
class RoBERTaClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(RoBERTaClassifier, self).__init__()

        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 9)
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        self.relu = nn.ReLU()
        
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

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer





class RoBERTa_RegHub(RoBERTaClassifier):
    def __init__(self,):            
        RoBERTaClassifier.__init__(self)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device=device
        self.to(self.device)
        
    def load_model(self,model_name='RoBERTa_classifier.pth',torch=torch,from_aws=False,bucket="fs-reghub-news-analysis"):
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
        self.lay=lay
        n=0
        for param in self.parameters():
            n=n+1
            param.requires_grad = False
            if n==(201-self.lay):
                break
   
    def hyper_parameters(self,epochs=100,LR=0.00001,criterion=CrossEntropyLoss(),optimizer=Adam):
        # hyperparameters
        self.epochs=epochs
        self.LR=LR
        self.criterion=criterion.to(self.device)
        self.optimizer=optimizer(self.parameters(), lr= self.LR)
                
              
    def pre_load(self, train_data=None, val_data=None, batch_size=32,Dataset=Dataset,DataLoader=DataLoader,tokenizer=tokenizer_RoBERTa):
        
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
            
            self.total_acc_train = 0
            self.total_loss_train = 0

            self.true_acc=0
            n=0
            
            with tqdm(self,total=len(self.train_dataloader), desc=f'Epoch {self.epoch_num + 1}/{self.epochs}', unit='item',position=0,leave=True) as p_bar:
                for train_input, train_label in self.train_dataloader:
                    train_label = train_label.to(self.device) # to cuda GPU
                    mask = train_input['attention_mask'].to(self.device) # attention mask
                    input_id = train_input['input_ids'].squeeze(1).to(self.device)

                    l1_loss=0
                    
                    
                    # for L1 regularization
                    a=0
                    reg_loss = 0
                    for param in self.parameters():
                        a=a+1
                        if a >=201-68:
                            reg_loss += torch.norm(param, 1)
                            
                            
                    # model output
                    output = self(input_id, mask)
                    
                    
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

        torch.save(checkpoint, f'RoBERTa_checkpoint_epoch{self.epoch_num+1}.pth')

        
        
        
    def save_best_model(self,save_aws=True,name='RoBERTa_classifier.pth'):
        # retrieve best checkpoint model
        self.name=name
        checkpoint = torch.load(f'RoBERTa_checkpoint_epoch{self.epoch_num+1-2}.pth')
        
        # self.load_state_dict(checkpoint['model_state_dict'])


        # save best model
        torch.save(checkpoint, self.name)
        # torch.save(self, self.name)
        
        # delete remaining model checkpoints
        for f in glob.glob("RoBERTa_checkpoint*.pth"):
            os.remove(f)
        
        if save_aws:
            self.save_to_aws()
    
    def save_to_aws(self,bucket="fs-reghub-news-analysis",awsOps=awsOps):
        with open("../aws_credentials.json", 'r') as file:
            aws_creds_json = json.load(file)
        
        aws = awsOps(aws_creds_json)
        aws.upload_file(bucket=bucket, path=self.name, name=self.name)
        
    
    
    def classifier(self,F=F,torch=torch,input_text="Commerzbank faced bankrupcy today",table_format=False,df_test=None,Dataset=Dataset,pd=pd,custom_model=None,tokenizer=tokenizer_RoBERTa):
        
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