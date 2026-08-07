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


def _run_and_capture(monkeypatch, user_prompt: str) -> tuple[dict, dict]:
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
        user_prompt=user_prompt,
        max_tokens=4096,
        enable_thinking=True,
        compact_prompt=False,
        max_attempts=1,
    )
    return captured["payload"], response.raw_response


def test_qwen35_explicit_thinking_gets_safe_default_budget(monkeypatch) -> None:
    payload, raw = _run_and_capture(monkeypatch, "ordinary non-structured prompt")

    assert payload["enable_thinking"] is True
    assert payload["thinking_budget"] == 1536
    assert payload["max_tokens"] == 4096
    assert raw["_llm4hls_thinking_budget"] == 1536
    assert raw["_llm4hls_generation_mode"] == "configured"


def test_structured_initial_generation_disables_thinking(monkeypatch) -> None:
    payload, raw = _run_and_capture(
        monkeypatch,
        "Structured exploration contract:\n- Strategy family: buffered_parallelism.",
    )

    assert payload["enable_thinking"] is False
    assert "thinking_budget" not in payload
    assert raw["_llm4hls_generation_mode"] == "direct_non_thinking"


def test_partial_corrective_retry_stays_non_thinking(monkeypatch) -> None:
    payload, raw = _run_and_capture(
        monkeypatch,
        "Structured exploration contract:\n"
        "CORRECTIVE RETRY FOR THIS SAME CANDIDATE SLOT:\n"
        "- The previous response was rejected before Vitis because: buffered_parallelism_not_realised.\n",
    )

    assert payload["enable_thinking"] is False
    assert "thinking_budget" not in payload
    assert raw["_llm4hls_generation_mode"] == "corrective_non_thinking"


def test_no_semantic_change_retry_escalates_reasoning(monkeypatch) -> None:
    payload, raw = _run_and_capture(
        monkeypatch,
        "Structured exploration contract:\n"
        "CORRECTIVE RETRY FOR THIS SAME CANDIDATE SLOT:\n"
        "- The previous response was rejected before Vitis because: no_semantic_change.\n",
    )

    assert payload["enable_thinking"] is True
    assert payload["thinking_budget"] == 1536
    assert raw["_llm4hls_generation_mode"] == "reasoning_escalation"


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
        user_prompt="ordinary non-structured prompt",
        max_tokens=4096,
        enable_thinking=True,
        thinking_budget=1024,
        compact_prompt=False,
        max_attempts=1,
    )

    assert captured["payload"]["thinking_budget"] == 1024
