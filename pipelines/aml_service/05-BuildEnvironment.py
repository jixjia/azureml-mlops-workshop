import json
import sys
from azureml.core import Workspace, Environment

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path='aml_config/aml_config.json', auth=cli_auth)

# load mlops config
with open('aml_config/config.json') as f:
    config = json.load(f)

try:
    ''' 
    register the Environment for modle auto-test & inference 
    '''
    # env = Environment.from_conda_specification(name=config['env_name'], file_path='environment_setup/conda_dependencies.yml')
    # env.register(workspace=ws)

    '''
    or reuse a built environment image (if available and no dependency has changed since last build) 
    so to avoid rebuilding container image each time
    '''
    env = Environment.get(workspace=ws, name=config['ENV_NAME'])

    print(f'[INFO] Successfully retrieved Environment {env.name}:{env.version}')

except Exception as e:
    print(e.args)
    sys.exit(0)