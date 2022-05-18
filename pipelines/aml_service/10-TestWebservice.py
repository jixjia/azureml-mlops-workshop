import requests
import os
import json
import sys
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()


# set workspace
ws = Workspace.from_config(path='aml_config/aml_config.json', auth=cli_auth)

# load mlops config
with open('aml_config/config.json') as f:
    config = json.load(f)

source_directory = config['SOURCE_DIRECTORY']
deploy_folder = config['DEPLOY_FOLDER']

# read deployed webservice detail
try:
    webservice_path = os.path.join(source_directory, deploy_folder, 'aci_webservice.json')

    with open(webservice_path) as f:
        aci_config = json.load(f)
except:
    print('[INFO] No webservice deployment log found. Exiting...')
    sys.exit(0)

# get the hosted webservice
service_name = aci_config['aci_service_name']

service = Webservice(name=service_name, workspace=ws)

# test webservice using sample image
try:
    payload = {
        'query': '',
        'template':''
    }

    inputs = json.dumps(payload)

    prediction = service.run(input_data=inputs)
    print(prediction)

except Exception as e:
    print(e.args)
    raise Exception(f'[INFO] Webservice not functioning. {e.args}')