from __future__ import annotations

import json

from agent.providers import siliconflow


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_qwen35_explicit_thinking_gets_safe_default_budget(monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "complete source"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )

    monkeypatch.setattr(siliconflow.urllib.request, "urlopen", fake_urlopen)

    response = siliconflow.complete(
        model="Qwen/Qwen3.5-122B-A10B",
        system_prompt="system",
        user_prompt="user",
        max_tokens=4096,
        enable_thinking=True,
        compact_prompt=False,
        max_attempts=1,
    )

    assert captured["payload"]["enable_thinking"] is True
    assert captured["payload"]["thinking_budget"] == 1536
    assert captured["payload"]["max_tokens"] == 4096
    assert response.raw_response["_llm4hls_thinking_budget"] == 1536


def test_explicit_thinking_budget_overrides_safe_default(monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "complete source"},
                    }
                ],
                "usage": {},
            }
        )

    monkeypatch.setattr(siliconflow.urllib.request, "urlopen", fake_urlopen)

    siliconflow.complete(
        model="Qwen/Qwen3.5-122B-A10B",
        system_prompt="system",
        user_prompt="user",
        max_tokens=4096,
        enable_thinking=True,
        thinking_budget=1024,
        compact_prompt=False,
        max_attempts=1,
    )

    assert captured["payload"]["thinking_budget"] == 1024
