"""
Script to create the weekly output, considering the new news content.
"""

# Import necessary libraries
import time

# Import local functions
from functions_weekly_analysis import *

start = time.now()
print("General script startet.")

# Import data
df_news = onedrive_download("https://1drv.ms/u/s!AoiE7xOoBAsngsglIrXS2_lWtDSw1w?e=leaCqP")
df_categories = onedrive_download("https://1drv.ms/u/s!AoiE7xOoBAsngsglIrXS2_lWtDSw1w?e=leaCqP")
df_prompts = onedrive_download("https://1drv.ms/u/s!AoiE7xOoBAsngsglIrXS2_lWtDSw1w?e=leaCqP")

# Create suiting rule based categories
df_news_rule_cat = create_rule_cats(df_news, df_categories)

# Create suiting BERT based categories
# XXXX

# For now, changes once we have BERT
df_news_cat = df_news_rule_cat

# Display some info to the user
middle = time.now()
print("Categorization finished, now prompting LLM.")
print("It took " + str(middle - start) + " seconds so far.")

# Splitting the df into the different categories
dfs_cat_split = split_cat_df(df_news_cat)

# Add columns with the category-based llm-outputs
# Llama2
for key, df in dfs_cat_split:
    prompts = create_prompt(df, df_prompts[key][0])
    answer_list = []
    for prompt in prompts:
        answer = llama2_base(prompt)
        answer_list.append(answer)
        df[f"{key}_llama"] = answer_list

# Some more df operations

# Maybe more?
# ChatGPT function or something

# Display some info to the user
end = time.now()
print("Prompting finished, it took " + str(end - middle) + " seconds.")

# Save resulting df or do anything else with it
# df_news_enhanced.to_csv("news_data_enhanced.csv")

# Take time and print success message
print("Results saved, the script took " + str(end - start) + " seconds in total.")