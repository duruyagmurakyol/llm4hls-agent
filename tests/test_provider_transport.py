from __future__ import annotations

import json

from agent.providers import siliconflow


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_openrouter_transport_uses_openrouter_key_and_headers(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": "complete source"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
        )

    monkeypatch.setenv("LLM4HLS_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setattr(siliconflow.urllib.request, "urlopen", fake_urlopen)

    response = siliconflow.complete(
        model="open/model",
        system_prompt="system",
        user_prompt="user",
        max_attempts=1,
        compact_prompt=False,
    )

    assert response.provider == "openrouter"
    assert response.total_tokens == 14
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-openrouter-key"
    assert captured["headers"]["Http-referer"] == "https://llm4hls.local"
    assert captured["headers"]["X-title"] == "LLM4HLS Track A"
    assert "enable_thinking" not in captured["payload"]
    assert "thinking_budget" not in captured["payload"]


def test_siliconflow_transport_preserves_thinking_controls(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": "complete source"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        )

    monkeypatch.setenv("LLM4HLS_PROVIDER", "siliconflow")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-siliconflow-key")
    monkeypatch.setattr(siliconflow.urllib.request, "urlopen", fake_urlopen)

    response = siliconflow.complete(
        model="other/model",
        system_prompt="system",
        user_prompt="user",
        thinking_budget=128,
        enable_thinking=True,
        max_attempts=1,
        compact_prompt=False,
    )

    assert response.provider == "siliconflow"
    assert captured["url"] == "https://api.siliconflow.com/v1/chat/completions"
    assert captured["payload"]["enable_thinking"] is True
    assert captured["payload"]["thinking_budget"] == 128
