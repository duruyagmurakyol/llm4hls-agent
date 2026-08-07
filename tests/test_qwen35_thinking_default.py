from agent.onboarding_safe import DEFAULT_MODEL


def test_default_qwen35_uses_thinking_mode() -> None:
    assert DEFAULT_MODEL["name"] == "Qwen/Qwen3.5-122B-A10B"
    assert DEFAULT_MODEL["enable_thinking"] is True
    assert DEFAULT_MODEL["max_tokens"] == 4096
