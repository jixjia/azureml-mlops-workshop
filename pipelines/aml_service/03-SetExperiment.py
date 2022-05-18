import json
import sys
from azureml.core import Workspace, Experiment

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path='aml_config/aml_config.json', auth=cli_auth)

# load mlops config
with open('aml_config/config.json') as f:
    config = json.load(f)

try:
    # set or create an Experiment for tracking
    experiment_name = config['EXPERIMENT_NAME']
    exp = Experiment(workspace=ws, name=experiment_name)

    print(f'[INFO] Successfully set experiment\n{exp}')

except Exception as e:
    print(e.args)
    sys.exit(0)