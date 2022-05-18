# ChungHua Telecom Workshop Part 2
Create AzureDevOps Pipelines to automate CI/CD of Azure ML

## Pre-requisites

1. Initiate a new AzrueDevOps project

2. `git init` and `git remote add` to Azure DevOps Repos


## Setup Service Principal

1. Create a new `Service Principal` and add as `Contributor` role to the Azure ML Workspace resource group

    `Azure AD > App Registration > New Registration > 3 Months Expiry`

2. Take a note of followings of the new Service Principal:

    | Key | Value |
    |:-- |:-- |
    |Application ID (Client ID) | XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXX |
    |Client Secret Value|  XXXXX~XXXXXXXX~XXXXXXXXXXXXXXXX |
    |Directory ID (tenant ID)| XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXX |


3. Take a note of following from `Azure Subscription`:

    | Key | Value |
    |:-- |:-- |
    |Subscription ID | XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXX |

## Setup Azure Key Vault	

1. Create a new Azure Key Vault and create secrets with following mappings

    | Secret Name | Mapped Azure Property |
    |:-- |:-- |
    |spidentity | Application ID (Client ID) |
    |spsecret|  Client Secret |
    |sptenant| Directory ID (tenant ID) |
    |subscriptionid| Subscription ID|
	

## Setup Pipelines Variable Group
1. Create a Pipeline Variable Group named `AzureKeyVaultSecrets`, refencing the above Azure Key Vault secrets

    `Pipelines > Library > New Variable Group`

2. Grant Permissions when Pipeline runs for the first time

    | Permission | Object|
    |:-- |:-- |
    |Grant | Pipeline Variable Groups |
    |Grant | Service connection to Azure Subscription |