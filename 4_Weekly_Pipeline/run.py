"""
Script to create the weekly output, considering the new news content.

Run while on Repository (root) level!
"""

# Import necessary libraries
import time
import json
import pandas as pd

# Import local functions
from functions import *

start = time.time()
print("General script startet.")

# Import data
# Access aws credentials from json file
with open("aws_credentials.json", 'r') as file:
    aws_creds_json = json.load(file)
# Specify s3 bucket
bucket = "fs-reghub-news-analysis"

# Connect to aws and dowload the files
aws = awsOps(aws_creds_json)
df_news = aws.get_df(bucket=bucket, file="data_raw_extended.csv")[:10]
df_categories = aws.get_df(bucket=bucket, file="rule_labels_v1.csv").drop(columns="Unnamed: 0")
df_prompts = aws.get_df(bucket=bucket, file="llama_prompts_v2.csv").drop(columns="Unnamed: 0")

# Create suiting rule based categories
df_news_rule_cat = create_rule_cats(df_news, df_categories)

# Create suiting BERT based categories
#df_news_bert_cat = create_bert_cats(df_news, df_categories)

# For now we continue with rule-based
df_news_cat = df_news_rule_cat
#df_news_cat = df_news_bert_cat

# Display some info to the user
middle = time.time()
print("Categorization finished, now prompting LLM.")
print("It took " + str(round(middle - start, 1)) + " seconds so far.")

# Splitting the df into the different categories
dfs_cat_split = split_cat_df(df_news_cat, df_categories)

# Add columns with category based Llama2
dfs_llm_split = create_llama2_cols(dfs_cat_split, df_prompts)

# Merge category dfs back with main df
df_news_llm = df_news_cat.copy()
for df in dfs_llm_split:
    df_news_llm = pd.merge(df_news_llm, df, on="id", how="left")

# Display some info to the user
end = time.time()
print("Prompting finished, it took " + str(round(end - middle, 1)) + " seconds.")

# Save resulting df
df_news_llm.to_csv("../data/data_llama_v1.csv")

# Upload the data to aws
# Access aws credentials from json file
with open("aws_credentials.json", 'r') as file:
    aws_creds_json = json.load(file)
# Specify s3 bucket
bucket = "fs-reghub-news-analysis"
# Connect to aws and upload the files
aws = awsOps(aws_creds_json)
print(aws.upload_file(bucket, "../data/data_llama_v1.csv", "data_llama_v1.csv"))

# Take time and print success message
print("Results saved, the script took " + str(round(end - start, 1)) + " seconds in total.")