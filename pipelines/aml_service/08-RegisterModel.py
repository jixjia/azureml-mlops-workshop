import os
import json
import sys
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.resource_configuration import ResourceConfiguration

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path='aml_config/aml_config.json', auth=cli_auth)

# load mlops config
with open('aml_config/config.json') as f:
    config = json.load(f)

model_name = config['MODEL_NAME']
source_directory = config['SOURCE_DIRECTORY']
test_folder = config['TEST_FOLDER']
model_dir = os.path.join(source_directory, test_folder, config['MODEL_PATH'])
model_description = config['MODEL_DESCRIPTION']

try:
    model = Model.register(workspace = ws,
                       model_name = model_name,
                       model_path = model_dir,
                       model_framework = Model.Framework.TENSORFLOW,
                       tags = {'created by': 'AzreDevOps', 'source': "CHT Workshop", 'type': "Custom Siamese Network"},
                       description = model_description,
                       resource_configuration = ResourceConfiguration(cpu=1, memory_in_gb=2)
                       )

    print(f'[INFO] Successfully registered the model {model.name} (version: {model.version})')

except Exception as e:
    print(e.args)
    sys.exit(0)