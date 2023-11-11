"""
Script to create the weekly output, considering the new news content.
"""

# Import necessary libraries
import time
import pandas as pd

# Import local functions
from functions_weekly_analysis import *

start = time.time()
print("General script startet.")

# Import data
df_news = pd.read_csv(onedrive_download("https://1drv.ms/u/s!AoiE7xOoBAsngsgTerpuaYKsupEgYA?e=ASsoss")).iloc[:25]
df_categories = pd.read_csv(onedrive_download("https://1drv.ms/u/s!AoiE7xOoBAsngsgkB0_f7WIAay63-Q?e=ApZRae"))
df_prompts = pd.read_csv(onedrive_download("https://1drv.ms/u/s!AoiE7xOoBAsngsgq8jdbX28to6_3zg?e=OIyfW6"))

# Create suiting rule based categories
df_news_rule_cat = create_rule_cats(df_news, df_categories)

# Create suiting BERT based categories
# XXXX

# For now, changes once we have BERT
df_news_cat = df_news_rule_cat

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

# Maybe more?
# ChatGPT function or something

# Display some info to the user
end = time.time()
print("Prompting finished, it took " + str(round(end - middle, 1)) + " seconds.")

# Save resulting df or do anything else with it
df_news_llm.to_csv("../news_data_enhanced_test.csv")

# Take time and print success message
print("Results saved, the script took " + str(round(end - start, 1)) + " seconds in total.")