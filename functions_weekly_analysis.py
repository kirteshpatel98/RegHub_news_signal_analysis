# Import necessary libraries
import base64
import pandas as pd
from langchain.llms import Ollama

# Define necessary functions
def onedrive_download(link):
    """
    Retrieve csvs from onedrive.
    """

    # What is happening?
    data = base64.b64encode(bytes(link, 'utf-8'))
    data_string = data.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    download_url = f"https://api.onedrive.com/v1.0/shares/u!{data_string}/root/content"
    
    return download_url

def create_rule_cats(df, cats):
    """
    Create rule-based categories.
    """

    # Transform different categories to lists
    pers_cat_words = list(cats["personnel"].dropna())
    prod_cat_words = list(cats["product"].dropna())
    colab_cat_words = list(cats["collaboration"].dropna())
    legal_cat_words = list(cats["legal"].dropna())

    # Add columns for rule-based labels
    df["rule_labels_pers"] = "Other"
    df["rule_labels_prod"] = "Other"
    df["rule_labels_colab"] = "Other"
    df["rule_labels_legal"] = "Other"

    # Match words from the category lists
    for index, row in df.iterrows():
        if any(word.lower() in row["news_content"].lower(
        ) for word in pers_cat_words):
            df.at[index, "rule_labels_pers"] = "personnel"
        if any(word.lower() in row["news_content"].lower(
        ) for word in prod_cat_words):
            df.at[index, "rule_labels_prod"] = "product"
        if any(word.lower() in row["news_content"].lower(
        ) for word in colab_cat_words):
            df.at[index, "rule_labels_colab"] = "collaboration"
        if any(word.lower() in row["news_content"].lower(
        ) for word in legal_cat_words):
            df.at[index, "rule_labels_legal"] = "legal"

    # Compress the four category columns into one
    df['rule_labels_comb'] = df[['rule_labels_pers', 'rule_labels_prod',
                                 'rule_labels_colab', 'rule_labels_legal']
                                 ].values.tolist()
    df['rule_labels_comb'] = df['rule_labels_comb'
    ].apply(lambda lst: [val for val in lst if val != 'Other'])
    df.drop(columns=["rule_labels_pers", "rule_labels_prod",
                     "rule_labels_colab", "rule_labels_legal"
                     ], inplace=True)
    
    return df

def split_cat_df(df, cats):
    """
    Description
    """

    categories = list(cats.columns)
    dfs_split = {}
    for category in categories:
        df_split = df[df['rule_labels_comb'].apply(lambda lst: category in lst)]
        dfs_split[category] = df_split

    return dfs_split

def create_llama2_cols(df_dict, df_prompts):
    """
    DDD
    """

    # Import model
    model = Ollama(base_url="http://localhost:11434", model="llama2")

    # Loop through the four category dfs, create and ask model
    dfs_llm_split = []
    for key, df in df_dict.items():
        df = df[["id", "news_content"]].copy()
        df[f"{key}_prompt"] = df["news_content"] + df_prompts[key][0]
        df[f"{key}_llama"] = ""
        # Result for each row
        df[f"{key}_llama"
        ] = df.apply(lambda row: model(row[f"{key}_prompt"]), axis=1)
        # Some reshaping and storing the results
        df.drop(columns="news_content", inplace=True)
        dfs_llm_split.append(df)
        
    return dfs_llm_split