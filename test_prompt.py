import json
import os
import urllib.error
import urllib.request

api_key = os.environ.get("SILICONFLOW_API_KEY")

if not api_key:
    raise SystemExit("SILICONFLOW_API_KEY is not set")

url = "https://api.siliconflow.com/v1/chat/completions"

payload = {
    "model": "moonshotai/Kimi-K2.7-Code",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an expert AMD Vitis HLS engineer. "
                "Return only complete synthesizable C++ code. "
                "Do not use Markdown code fences. "
                "Do not invent headers, interfaces, functions or pragmas. "
                "Ensure the returned code is complete and not truncated."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a complete Vitis HLS function with this exact signature:\n"
                "int add(int a, int b)\n\n"
                "The function must return a + b. Return only the complete code."
            ),
        },
    ],
    "max_tokens": 2048,
    "temperature": 0.1,
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)

    print(data["choices"][0]["message"]["content"])

except urllib.error.HTTPError as error:
    print(f"HTTP error: {error.code}")
    print(error.read().decode())

except Exception as error:
    print(f"Request failed: {error}")
