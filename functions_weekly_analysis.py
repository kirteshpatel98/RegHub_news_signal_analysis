# Import necessary libraries
import base64
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

def llama2_base(prompt):
    """
    Function to specify llama model.
    """

    # Import the local model (ollama distribution)
    ollama = Ollama(base_url="http://localhost:11434", model="llama2")

    # Prompt the model
    answer = ollama(prompt)

    return answer