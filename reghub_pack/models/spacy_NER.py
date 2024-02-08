import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import spacy
import seaborn as sns

import os
import base64
import gc
import glob, os

import warnings
warnings.filterwarnings("ignore")

# Import local functions
from general_functions import *

# Import functions
from collections import Counter
from spacy import displacy

import json
from ..Dataset import *
from ..general_functions import *
from ..tokenizer import tokenizer_BERT

'''
!python -m spacy download en_core_web_lg
!python -m spacy download de_core_web_lg
'''


class spaCy_NER_en:
    def __init__(self,nlp_en = spacy.load("en_core_web_lg")):
        self.nlp_en = nlp_en
    
    def run_NER(self,text):
        print(displacy.render(text, style='ent'))
        
        

class spaCy_NER_de:
    def __init__(self,nlp_en = spacy.load("de_core_web_lg")):
        self.nlp_en = nlp_en
    
    def run_NER(self,text):
        print(displacy.render(text, style='ent'))
        
    