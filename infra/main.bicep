// ============================================================
// Zero Trading Agent — Azure Infrastructure
// Subscription: 97cbd3af-ec60-4824-8f28-e02ccb584ff3
// Region: Central India
// ============================================================

@description('Azure region for all resources')
param location string = 'centralindia'

@description('Base name for resources')
param baseName string = 'zero'

@description('Container image (updated by CI/CD)')
param containerImage string = 'mcr.microsoft.com/hello-world:latest'

// ============================================================
// Variables
// ============================================================
var acrName = 'acr${baseName}trading'
var keyVaultName = 'kv-${baseName}-trading'
var storageAccountName = 'st${baseName}trading'
var fileShareName = '${baseName}-state'
var logAnalyticsName = 'law-${baseName}-trading'
var containerEnvName = 'cae-${baseName}-trading'
var containerAppName = 'ca-${baseName}-agent'
var authJobName = 'ca-${baseName}-auth'
var dashboardAppName = 'ca-${baseName}-dashboard'

// ============================================================
// Azure Container Registry (Basic SKU)
// ============================================================
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// ============================================================
// Azure Key Vault
// ============================================================
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
  }
}

// ============================================================
// Storage Account + Azure Files Share
// ============================================================
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-01-01' = {
  parent: fileService
  name: fileShareName
  properties: {
    shareQuota: 5
    accessTier: 'Hot'
  }
}

// ============================================================
// Log Analytics Workspace
// ============================================================
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ============================================================
// Container Apps Environment
// ============================================================
resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Azure Files storage mount on the environment
resource envStorage 'Microsoft.App/managedEnvironments/storages@2023-05-01' = {
  parent: containerEnv
  name: fileShareName
  properties: {
    azureFile: {
      accountName: storageAccount.name
      accountKey: storageAccount.listKeys().keys[0].value
      shareName: fileShareName
      accessMode: 'ReadWrite'
    }
  }
}

// ============================================================
// Container App — Trading Agent (KEDA cron scaled)
// ============================================================
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  dependsOn: [envStorage]
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
      ingress: {
        external: false
        targetPort: 8080
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'zero-agent'
          image: containerImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'AZURE_DEPLOYMENT', value: 'true' }
            { name: 'PAPER_TRADE', value: 'true' }
            { name: 'STATE_DIR', value: '/state' }
            { name: 'AZURE_KEY_VAULT_URL', value: keyVault.properties.vaultUri }
          ]
          volumeMounts: [
            {
              volumeName: 'state-volume'
              mountPath: '/state'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/healthz'
                port: 8080
              }
              initialDelaySeconds: 30
              periodSeconds: 60
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'state-volume'
          storageName: fileShareName
          storageType: 'AzureFile'
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'market-hours-cron'
            custom: {
              type: 'cron'
              metadata: {
                timezone: 'Asia/Kolkata'
                start: '45 8 * * 1-5'
                end: '50 15 * * 1-5'
                desiredReplicas: '1'
              }
            }
          }
        ]
      }
    }
  }
}

// ============================================================
// Container App Job — Daily Auth (Playwright headless login)
// ============================================================
resource authJob 'Microsoft.App/jobs@2023-05-01' = {
  name: authJobName
  location: location
  dependsOn: [envStorage]
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: containerEnv.id
    configuration: {
      triggerType: 'Schedule'
      scheduleTriggerConfig: {
        cronExpression: '40 8 * * 1-5'
        parallelism: 1
        replicaCompletionCount: 1
      }
      replicaTimeout: 300
      replicaRetryLimit: 2
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'zero-auth'
          image: containerImage
          command: [ 'python', '-m', 'src.utils.headless_auth' ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'AZURE_DEPLOYMENT', value: 'true' }
            { name: 'STATE_DIR', value: '/state' }
            { name: 'AZURE_KEY_VAULT_URL', value: keyVault.properties.vaultUri }
          ]
          volumeMounts: [
            {
              volumeName: 'state-volume'
              mountPath: '/state'
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'state-volume'
          storageName: fileShareName
          storageType: 'AzureFile'
        }
      ]
    }
  }
}

// ============================================================
// Container App — Dashboard (optional, external ingress)
// ============================================================
resource dashboardApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: dashboardAppName
  location: location
  dependsOn: [envStorage]
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
      ingress: {
        external: true
        targetPort: 8501
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'zero-dashboard'
          image: containerImage
          command: [ 'streamlit', 'run', 'dashboard/app.py', '--server.port=8501', '--server.headless=true' ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            { name: 'STATE_DIR', value: '/state' }
          ]
          volumeMounts: [
            {
              volumeName: 'state-volume'
              mountPath: '/state'
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'state-volume'
          storageName: fileShareName
          storageType: 'AzureFile'
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

// ============================================================
// RBAC: Key Vault Secrets User for Container App + Auth Job
// ============================================================
@description('Key Vault Secrets User role')
var keyVaultSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

resource kvRoleAssignmentApp 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, containerApp.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyVaultSecretsUserRoleId)
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource kvRoleAssignmentJob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, authJob.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyVaultSecretsUserRoleId)
    principalId: authJob.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// ============================================================
// Outputs
// ============================================================
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
output containerAppName string = containerApp.name
output authJobName string = authJob.name
output dashboardUrl string = 'https://${dashboardApp.properties.configuration.ingress.fqdn}'
output containerEnvName string = containerEnv.name
output storageAccountName string = storageAccount.name
