#!/usr/bin/env python3

"""Fail-fast availability check for every model in an experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.providers.siliconflow import complete  # noqa: E402


def _load_suite(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read suite definition {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise RuntimeError(f"Suite definition has no models list: {path}")
    return value


def _required_key(provider: str) -> str:
    return "OPENROUTER_API_KEY" if provider == "openrouter" else "SILICONFLOW_API_KEY"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that every configured matrix model accepts a minimal request."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=REPO_ROOT / "configs" / "suites" / "overnight_60.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "model_preflight.json",
    )
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()

    suite_path = args.suite.expanduser().resolve()
    suite = _load_suite(suite_path)
    results: list[dict[str, Any]] = []

    for raw in suite["models"]:
        if not isinstance(raw, dict):
            continue
        model_id = str(raw.get("id", "")).strip()
        provider = str(raw.get("provider", "siliconflow")).strip().casefold()
        slug = str(raw.get("slug", model_id)).strip()
        key_name = _required_key(provider)
        print(f"Checking {model_id} via {provider}...", flush=True)

        record: dict[str, Any] = {
            "model_id": model_id,
            "model_slug": slug,
            "provider": provider,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "available": False,
            "input_tokens": None,
            "output_tokens": None,
            "latency_seconds": None,
            "response_preview": None,
            "error": None,
        }

        if not os.environ.get(key_name):
            record["error"] = f"missing {key_name}"
            print(f"  FAIL: {record['error']}", flush=True)
            results.append(record)
            continue

        previous_provider = os.environ.get("LLM4HLS_PROVIDER")
        os.environ["LLM4HLS_PROVIDER"] = provider
        try:
            response = complete(
                model=model_id,
                system_prompt="Return a short plain-text response.",
                user_prompt="Reply with exactly OK.",
                temperature=0.0,
                max_tokens=32,
                timeout_seconds=args.timeout_seconds,
                max_attempts=1,
                enable_thinking=False,
                compact_prompt=False,
                provider=provider,
            )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"  FAIL: {record['error']}", flush=True)
        else:
            record.update(
                {
                    "available": True,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_seconds": response.latency_seconds,
                    "response_preview": response.content.strip()[:120],
                }
            )
            print(
                "  PASS: "
                f"response={record['response_preview']!r} "
                f"tokens={response.total_tokens} "
                f"latency={response.latency_seconds:.2f}s",
                flush=True,
            )
        finally:
            if previous_provider is None:
                os.environ.pop("LLM4HLS_PROVIDER", None)
            else:
                os.environ["LLM4HLS_PROVIDER"] = previous_provider
        results.append(record)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "suite": str(suite_path),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_available": all(item["available"] is True for item in results),
        "models": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}")
    raise SystemExit(0 if payload["all_available"] else 1)


if __name__ == "__main__":
    main()
