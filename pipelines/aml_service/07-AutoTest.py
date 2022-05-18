import os
import json
import sys
from azureml.core import Workspace, Experiment, Datastore, Dataset, ScriptRunConfig, Environment

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path='aml_config/aml_config.json', auth=cli_auth)

# load mlops config
with open('aml_config/config.json') as f:
    config = json.load(f)

aml_pipeline_name = 'build-pipeline'
experiment_name = config['EXPERIMENT_NAME']
aml_cluster_name = config['COMPUTE_CLUSTER_NAME']
source_directory = config['SOURCE_DIRECTORY']
environment_name = config['ENV_NAME']
model_path = config['MODEL_PATH']
test_folder = config['TEST_FOLDER']
test_path = os.path.join(source_directory, test_folder)

exp = Experiment(workspace=ws, name=experiment_name)
compute_target = ws.compute_targets[aml_cluster_name]
dataset_test = Dataset.get_by_name(workspace=ws, name=config['DATASET_TEST']) 
env = Environment.get(workspace=ws, name=environment_name)

# mount test dataset to compute target 
mnt_test = dataset_test.as_mount()

# auto-test job config
src = ScriptRunConfig(source_directory = test_path,
                      script = 'test.py',
                      arguments = ['--mnt_path', mnt_test, 
                                   '--test_input', 'mario',
                                   '--model_path', model_path],
                      compute_target = compute_target,
                      environment = env)

# submit job and start tracking
run = exp.submit(config=src)

# sync submission
run.wait_for_completion()