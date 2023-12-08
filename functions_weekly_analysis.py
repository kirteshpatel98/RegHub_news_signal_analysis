# Import necessary libraries
import base64
import boto3
import pandas as pd
from io import BytesIO
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