from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, AmlCompute
from azure.core.exceptions import ResourceNotFoundError

# --- 1. CONNECT TO AZURE ML WORKSPACE ---
try:
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",       
        resource_group_name="rg-60304711",    
        workspace_name="Amazon-Electronics-Lab-60304711"          
    )
    print(f"Connected to Workspace: {ml_client.workspace_name}")
except Exception as ex:
    print("Failed to connect to Azure ML Workspace. Did you run 'az login'?")
    raise ex

# --- 2. PROVISION SCALABLE COMPUTE ---
compute_name = "cpu-cluster-lab5"
try:
    cpu_cluster = ml_client.compute.get(compute_name)
    print(f"Found existing compute cluster: {compute_name}")
except ResourceNotFoundError:
    print("Creating new scalable compute cluster...")
    cpu_cluster = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="STANDARD_DS3_V2", 
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=120,
    )
    ml_client.compute.begin_create_or_update(cpu_cluster).result()
    print("Compute cluster created.")

# --- 3. DEFINE REPRODUCIBLE CLOUD ENVIRONMENT ---
print("Configuring Cloud Environment...")
pipeline_env = Environment(
    name="lab5_final_env",  # Changed the name to force a new build
    version="1",            # Added a version number
    description="Fresh environment for Lab 5",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file="env.yml"
)

# --- 4. DEFINE AND SUBMIT THE PIPELINE JOB ---
print("Configuring Command Job...")
job = command(
    code="./",  
    command="python pipeline.py",
    environment=pipeline_env,
    compute=compute_name,
    experiment_name="lab5_feature_extraction_pipeline",
    display_name="NASA_CMAPSS_Feature_Pipeline"
)

print("Submitting job to Azure ML...")
returned_job = ml_client.jobs.create_or_update(job)
print(f" Job submitted successfully!")
print(f"Track your pipeline run here: {returned_job.studio_url}")