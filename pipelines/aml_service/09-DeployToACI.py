import os
import json
import datetime
import uuid
import sys
from azureml.core import Workspace, Environment
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.environment import Environment

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
environment_name = config['ENV_NAME']
model_name = config['MODEL_NAME']
model_description = config['MODEL_DESCRIPTION']
source_directory = config['SOURCE_DIRECTORY']
deploy_folder = config['DEPLOY_FOLDER']

env = Environment.get(workspace=ws, name=environment_name)

# container host
aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 2, 
                                               tags = {'model':model_name, 'framework':environment_name}, 
                                               description = model_description)

# inference config
inference_config = InferenceConfig(entry_script = 'score.py', 
                                   source_directory = os.path.join(source_directory, deploy_folder), 
                                   environment = env,
                                   enable_gpu = False)

# retrieve registered model by name
model = Model(workspace=ws, name=model_name)

# container host name (using model version as suffix)
service_name = f'{model_name}-v{model.version}'

try:
    service = Model.deploy(workspace = ws, 
                        name = service_name, 
                        models = [model], 
                        inference_config = inference_config, 
                        deployment_config = aci_config,
                        overwrite = True)

    # provision & deploy
    service.wait_for_deployment(show_output=True)

    print(f'Deployed ACI Webservice: {service.name} \nWebservice Uri: {service.scoring_uri}')

    # write endpoint detail to local for unit testing in next step
    aci_webservice = {}
    aci_webservice['aci_service_name'] = service.name
    aci_webservice['aci_service_edndpoint'] = service.scoring_uri
    webservice_path = os.path.join(source_directory, deploy_folder, 'aci_webservice.json')

    with open(webservice_path, 'w') as f:
        json.dump(aci_webservice, f)

except Exception as e:
    print(e.args)
    sys.exit(0)