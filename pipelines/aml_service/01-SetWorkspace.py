import json
from azureml.core import Workspace

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

with open('aml_config/aml_config.json') as f:
    config = json.load(f)

workspace_name = config['workspace_name']
resource_group = config['resource_group']
subscription_id = config['subscription_id']
location = config['location']

try:
    # set workspace
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth,
    )

except:
    # if doesn't exist then create one, could take 2-3 min
    print(f'[INFO] {workspace_name} does not exist, creating it...')
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        create_resource_group=True,
        location=location,
        auth=cli_auth,
    )

print('[INFO] Successfully retrieved workspace:', ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
