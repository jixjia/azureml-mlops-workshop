import json, os
from azureml.core import Workspace, Experiment, Run

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path='aml_config/aml_config.json', auth=cli_auth)

# load mlops config
with open('aml_config/config.json') as f:
    config = json.load(f)

experiment_name = config['EXPERIMENT_NAME']
model_path = config['MODEL_PATH']
source_directory = config['SOURCE_DIRECTORY']
test_path = os.path.join(source_directory, config['TEST_FOLDER'])

exp = Experiment(workspace=ws, name=experiment_name)

for run in exp.get_runs(include_children=False):
    if run.status == 'Completed':
        run_id = run.id

        # download Run tracked training artifacts
        run.download_files(prefix=model_path, output_directory=test_path)
        print(f'[INFO] Downloaded latest model artifacts and test dataset to {test_path}')
        
        break

