import json
import os
import urllib.error
import urllib.request

api_key = os.environ.get("SILICONFLOW_API_KEY")

if not api_key:
    raise SystemExit("SILICONFLOW_API_KEY is not set")

url = "https://api.siliconflow.com/v1/models"

request = urllib.request.Request(
    url,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)

    print("Key works. Available models:\n")

    for model in data.get("data", [])[:20]:
        print(model.get("id"))

except urllib.error.HTTPError as error:
    print(f"HTTP error: {error.code}")
    print(error.read().decode())

except Exception as error:
    print(f"Request failed: {error}")
