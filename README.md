# RegHub Competitor News Analysis (Banking Sector)

## Overview
We are data science master students at Frankfurt School of Finance and Management. For this project we teamed up with RegHub (https://www.reghub.io) to investigate how public news about competitors in the german banking space can be analysed on an ongoing basis. We worked with a dataset of 14609 news articles to train and test our approaches. 

## Descripton of folders
### 1_Exploratoy_Data_Analysis
General exploration of the dataset. Visualization of various metrics including entity regocnition, ... and ...
### 2_Data_Preprocessing
Rule-based labelling of the news articles into one or more of the following categories:
- legal
- sanctions
- papers
- reports
- statements
- guidelines
- press
- personnel
- market

### 3_Modelling
Testing various supervised deep learning algorithms, using the rule-based labels for training and testing. We settled on BERT as our primary model for categorization.
Next to categorization we also include a short summary of the relevant events within the category. To generate this summary, Llama2 was used.
Alongside as part of information retrieval from the news articles we also used name entity recognision models to extract the name entities.
Also a similarity analysis was performed by training BERT-MLM, which can be used to filter out duplicate news articles.
### reghub_pack
This folder is used to package the models, to be able to release them as a pip package later on.

## Usage
The file run_weekly_analysis.py can be used to analyse the additional news articles of every week. It runs through the whole pipeline of BERT categorization and Llama2 summarization.

## Examples and results
Some screenshot of BERT categorization etc.
