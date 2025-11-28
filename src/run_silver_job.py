from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

extract_features_comp = ml_client.components.get(name="extract_features_component", version="6")

input_data = Input(type="uri_folder", path="azureml:tumor_images_raw:2")

job = extract_features_comp(input_data=input_data)
job.compute = "lab5-cluster"
job.environment = extract_features_comp.environment

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
