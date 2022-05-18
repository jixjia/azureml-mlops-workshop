import sys, os

print(f'[INFO] Checking Azure ML dependencies...')

try:

    from azureml.core import Workspace, Dataset, Datastore, Environment, Experiment, Run, Model, ScriptRunConfig, VERSION
    from azureml.core.webservice import AciWebservice
    from azureml.core.conda_dependencies import CondaDependencies
    from azureml.core.authentication import AzureCliAuthentication

    if float('.'.join(VERSION.split('.')[:2])) < 1.39:
        print(f'[INFO] Current Azure ML Python SDK version {VERSION}. Require 1.39+. Exiting...')
        sys.exit(0)
    else:
        print('[INFO] All Good!')

except Exception as e:
    print(e.args)