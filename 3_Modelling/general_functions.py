# Import necessary libraries
import base64
import boto3
import pandas as pd
from io import BytesIO

# Define necessary functions
def onedrive_download(link):
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