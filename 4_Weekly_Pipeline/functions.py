"""
Functions needed for the corresponding run-file.
"""

# Import necessary libraries
import sys
import boto3
import pandas as pd
from io import BytesIO
from langchain.llms import Ollama

# Define necessary functions
class awsOps:
    def __init__(self, creds, service="s3", region="eu-central-1"):
        self.aws_creds = {
            'service_name': service,
            'aws_access_key_id': creds["aws_access_key_id"],
            'aws_secret_access_key': creds["aws_secret_access_key"],
            'region_name': region
        }
        self.s3 = boto3.client(**self.aws_creds)

    def download_file(self, bucket, file, output):
        self.s3.download_file(bucket, file, output)
        return "File downloaded"

    def upload_file(self, bucket, path, name):
        self.s3.upload_file(path, bucket, name)
        return "File uploaded"
    
    def delete_file(self, bucket, file):
        self.s3.delete_object(Bucket=bucket, Key=file)
        return "File deleted"

    def get_df(self, bucket, file):
        response = self.s3.get_object(Bucket=bucket, Key=file)
        file_content = response['Body'].read()
        csv_object = BytesIO(file_content)
        df = pd.read_csv(csv_object)
        return df

    def list_bucket_files(self, bucket):
        s3_resource = boto3.resource(**self.aws_creds)
        bucket = s3_resource.Bucket(bucket)
        files = [obj.key for obj in bucket.objects.all()]
        return files

def create_rule_cats(df, df_categories):
    """
    Create rule-based categories.
    """

    # Add spaces before and after the strings to avoid 
    # partially matching wrong words with short keys
    df_categories = df_categories.map(lambda x: f' {x} ')

    # Clean the df
    df.dropna(subset=["news_content"], inplace=True)

    # Create a list of the newly created columns for later use
    rule_cols = []
    # Create rule based category labels
    for category in df_categories.columns:
        # Keep track of which keywords led to categorization
        df_categories[f"{category}_count"] = 0
        # Add columns for rule-based labels
        df[f"rule_labels_{category}"] = "Other"
        rule_cols.append(f"rule_labels_{category}")
        # Match words from the category lists
        for index, row in df.iterrows():
            for word in list(df_categories[category].dropna()):
                if word.lower() in str(row["news_content"]).lower():
                    df.at[index, f"rule_labels_{category}"] = category
                    df_categories.loc[df_categories[category] == word, f"{category}_count"] += 1

    # Compress the nine category columns into one
    df['rule_labels_comb'] = df[rule_cols].values.tolist()
    df['rule_labels_comb'] = df['rule_labels_comb'].apply(lambda lst: [val for val in lst if val != 'Other'])
    df.drop(columns=rule_cols, inplace=True)
    
    return df

def create_bert_cats(df, df_categories):
    """
    Categorization using the BERT model.
    """

    # Import the BERT model
    sys.path.append("../RegHub_news_signal_analysis")
    # Add the parent directory to the Python path
    from reghub_pack.models.BERT import BERT_RegHub
    model = BERT_RegHub()
    model.load_model(from_aws=True)
    
    # Clean the df
    df.dropna(subset=["news_content"], inplace=True)
    
    # Create a list of the newly created columns for later use
    bert_cols = []
    # Create bert based category labels
    for category in df_categories.columns:
            # Add columns for bert-based labels
        df[f"bert_labels_{category}"] = "Other"
        bert_cols.append(f"bert_labels_{category}")
        # Find suiting categories using bert
        for index, row in df.iterrows():
            bert_out = model.classifier(input_text=str(row["news_content"]))

            # Work with bert dictionary output
            max_key = max(bert_out, key=bert_out.get)
            if max_key > 50:
                df.at[index, f"bert_labels_{max_key}"] = max_key
            else:
                # Find highest
                first_max_key = max(bert_out, key=bert_out.get)
                bert_out.pop(max_key)
                # And second highest category
                second_max_key = max(bert_out, key=bert_out.get)
                # Add to df
                df.at[index, f"bert_labels_{first_max_key}"] = first_max_key
                df.at[index, f"bert_labels_{second_max_key}"] = second_max_key

    # Compress the nine category columns into one
    df['bert_labels_comb'] = df[bert_cols].values.tolist()
    df['bert_labels_comb'] = df['bert_labels_comb'].apply(lambda lst: [val for val in lst if val != 'Other'])
    df.drop(columns=bert_cols, inplace=True)

    return df

def split_cat_df(df, cats):
    """
    Splitting the df into the different categories
    to run them through different individual 
    Llama2 prompts later.
    """

    categories = list(cats.columns)
    dfs_split = {}
    for category in categories:
        df_split = df[df['rule_labels_comb'].apply(lambda lst: category in lst)]
        if not df_split.empty:
            dfs_split[category] = df_split

    return dfs_split

def create_llama2_cols(df_dict, df_prompts):
    """
    Create the Llama2 based summaries based on the given category.
    """

    # Import model
    model = Ollama(base_url="http://localhost:11434", model="llama2")

    # Loop through the four category dfs, create and ask model
    dfs_llm_split = []
    for key, df in df_dict.items():
        df = df[["id", "news_content"]].copy()
        prompts = []
        for index, row in df.iterrows():
            prompt = df_prompts[key][0
                    ].format(article=row["news_content"])
            prompts.append(prompt)
        df[f"{key}_prompt"] = prompts
        df[f"{key}_llama"] = ""
        # Result for each row
        df[f"{key}_llama"
        ] = df.apply(lambda row: model(row[f"{key}_prompt"]), axis=1)
        # Some reshaping and storing the results
        df.drop(columns="news_content", inplace=True)
        dfs_llm_split.append(df)
        
    return dfs_llm_split