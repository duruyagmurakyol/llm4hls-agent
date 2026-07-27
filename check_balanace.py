import json
import os
import sys
import urllib.error
import urllib.request

api_key = os.environ.get("SILICONFLOW_API_KEY")

if not api_key:
    print("ERROR: SILICONFLOW_API_KEY is not set")
    sys.exit(1)

request = urllib.request.Request(
    "https://api.siliconflow.com/v1/user/info",
    headers={"Authorization": f"Bearer {api_key}"},
    method="GET",
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    account = data["data"]

    print("Remaining balances:")
    print("Gifted balance:", account.get("balance"))
    print("Recharged balance:", account.get("chargeBalance"))
    print("Total balance:", account.get("totalBalance"))
    print("Account status:", account.get("status"))

except urllib.error.HTTPError as error:
    print("HTTP error:", error.code)
    print(error.read().decode("utf-8"))

except Exception as error:
    print("Error:", error)
