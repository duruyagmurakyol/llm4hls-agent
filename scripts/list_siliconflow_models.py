#!/usr/bin/env python3

"""List SiliconFlow models matching competition-relevant name fragments."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


KEY_NAMES = ("SILICONFLOW_API_KEY", "SILICONFLOW_KEY")
DEFAULT_BASE_URL = "https://api.siliconflow.com/v1"


def main() -> None:
    key = next((os.environ.get(name) for name in KEY_NAMES if os.environ.get(name)), None)
    if not key:
        raise SystemExit("Missing SILICONFLOW_API_KEY (or SILICONFLOW_KEY).")

    base_url = os.environ.get("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"SiliconFlow HTTP {error.code}: {body}") from error

    models = payload.get("data") or []
    needles = ("deepseek-v4", "qwen3.5", "qwen3.6")
    matches = sorted(
        str(item.get("id"))
        for item in models
        if isinstance(item, dict)
        and any(needle in str(item.get("id", "")).lower() for needle in needles)
    )
    if not matches:
        print("No exact competition-name matches found. Available DeepSeek/Qwen models:")
        matches = sorted(
            str(item.get("id"))
            for item in models
            if isinstance(item, dict)
            and any(token in str(item.get("id", "")).lower() for token in ("deepseek", "qwen"))
        )
    for model in matches:
        print(model)


if __name__ == "__main__":
    main()
