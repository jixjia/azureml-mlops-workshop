import json
import os
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path='aml_config/aml_config.json', auth=cli_auth)

# load mlops config
with open('aml_config/config.json') as f:
    config = json.load(f)

compute_cluster_name = config['COMPUTE_CLUSTER_NAME']

# verify that compute cluster exists already
try:
    compute_target = ComputeTarget(ws, compute_cluster_name)
    
    if compute_target and type(compute_target) is AmlCompute:
        print(f'[INFO] Set compute target to the specified clsuter: {compute_cluster_name}')

except ComputeTargetException:
    print(f'[INFO] Creating new compute target...')
    
    compute_min_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MIN_NODES', 0)
    compute_max_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MAX_NODES', 8)
    vm_size = os.environ.get('AML_COMPUTE_CLUSTER_SKU', 'STANDARD_NC6')

    provision_config = AmlCompute.provisioning_configuration(
                                                            vm_size = vm_size,
                                                            min_nodes = compute_min_nodes, 
                                                            max_nodes = compute_max_nodes,
                                                            vm_priority='dedicated',
                                                            idle_seconds_before_scaledown = '600'
                                                        )
    
    # create compute target
    compute_target = ComputeTarget.create(ws, compute_cluster_name, provision_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    # for a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())
