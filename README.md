# RegHub Competitor News Analysis (Banking Sector)

## Overview
We are data science master students at Frankfurt School of Finance and Management. For this project we teamed up with RegHub (https://www.reghub.io) to investigate how public news about competitors in the german banking space can be analysed on an ongoing basis. We worked with a dataset of 14609 news articles to train and test our approaches. 

## Descripton of folders
### 1_Exploratoy_Data_Analysis
General exploration of the dataset and various visualizations.
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
### 4_Weekly_Pipeline
Script that runs through the whole process of categorization and llama2 summary creation for a given dataset. Can be run at specified intervals to cover new news articles.
### 5_Misc
Collection of scripts that where used and or tested in the course of this project, but don't belong into the final main folders.
### reghub_pack
This folder is used to package the models, to be able to release them as a pip package later on.

## Usage
The run file in the weekly analysis folder can be used to analyse the additional news articles of every week. It runs through the whole pipeline of BERT categorization and Llama2 summarization.

## Examples and results
See the presentation of this project: https://1drv.ms/b/s!AsfpqRPTBA6DvH3pPeYWj7ub28lM?e=BIp20h

## Instructions to use reghub wrapper package
### Using PiP
### Step I
```
pip install reghub-pack
```

### Step II
```
import reghub_pack
```

### If the above method doesn't work, try using poetry
### Step I
Install dependent packages: (terminal)
```
conda env create -n myenv -f reghub_packages.yml
```     
### Step II
Clone repository branch: (terminal)
```
git clone -b reghub_pack https://github.com/kirteshpatel98/RegHub_news_signal_analysis
```

### Step III
Change directory to the clone repo
```
cd */RegHub_news_signal_analysis
```

### Step IV
Install poetry
```
pip install poetry
```

### Step V
Build poetry
```
poetry build
```

### Step VI
Add package to your environment
```
pip install .
```

### Step VII
Import package in python
```
import reghub_pack
```


## Acknowledgment

We would like to express our sincere gratitude to **Gerrit Knippschild** and **RegHub** for their invaluable assistance and support throughout the duration of our project. Their contribution in providing us with access to the dataset and offering expert guidance has been instrumental.
