from agent.onboarding_safe import DEFAULT_MODEL


def test_default_qwen35_uses_bounded_source_budget() -> None:
    assert DEFAULT_MODEL["name"] == "Qwen/Qwen3.5-122B-A10B"
    # Non-structured callers may still opt into thinking; structured C1-C3
    # generation is overridden to cheap direct generation at the provider.
    assert DEFAULT_MODEL["enable_thinking"] is True
    assert DEFAULT_MODEL["max_tokens"] == 4096
