import os
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_dataset_path", type=str, default="./assignment/data/brain_tumor_dataset")  # Your unzipped path
parser.add_argument("--storage_account_name", type=str, default="yourstorageaccount")  # Replace with your ADLS Gen2 account name
parser.add_argument("--container_name", type=str, default="raw")
args = parser.parse_args()

credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url=f"https://{args.storage_account_name}.blob.core.windows.net", credential=credential)
container_client = blob_service_client.get_container_client(args.container_name)

# Create container if not exists (idempotent)
try:
    container_client.create_container()
except Exception:
    pass

# Upload function (skip if exists)
def upload_if_not_exists(local_path, blob_path):
    blob_client = container_client.get_blob_client(blob_path)
    if not blob_client.exists():
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data)

# Upload yes/no folders
for folder in ['yes', 'no']:
    local_folder = os.path.join(args.local_dataset_path, folder)
    for file_name in os.listdir(local_folder):
        local_file = os.path.join(local_folder, file_name)
        blob_path = f"tumor_images/{folder}/{file_name}"
        upload_if_not_exists(local_file, blob_path)

print("Upload complete. Idempotent - skipped existing files.")
