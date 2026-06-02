"""Fetch container logs from Log Analytics API directly."""
import subprocess
import json
import os
import urllib3
urllib3.disable_warnings()
import requests

# Get Azure access token
result = subprocess.run(
    'az account get-access-token --resource https://api.loganalytics.io --query accessToken -o tsv',
    capture_output=True, text=True, shell=True
)
token = result.stdout.strip()

if not token:
    print("ERROR: Could not get access token")
    print(result.stderr)
    exit(1)

workspace_id = "5f991444-8182-4922-8af0-4583de84f9bc"
url = f"https://api.loganalytics.io/v1/workspaces/{workspace_id}/query"

query = """ContainerAppConsoleLogs_CL 
| where TimeGenerated > ago(1h) 
| project TimeGenerated, ContainerAppName_s, Log_s 
| order by TimeGenerated desc 
| take 50"""

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
r = requests.post(url, json={"query": query}, headers=headers, verify=False)
data = r.json()

if "tables" in data:
    rows = data["tables"][0]["rows"]
    if not rows:
        print("No logs found in last 1 hour")
    for row in rows:
        ts = row[0][:19] if row[0] else ""
        app = row[1] or ""
        log = (row[2] or "").replace('\n', ' ')[:150]
        print(f"{ts} | {app} | {log}")
else:
    print(json.dumps(data, indent=2))
