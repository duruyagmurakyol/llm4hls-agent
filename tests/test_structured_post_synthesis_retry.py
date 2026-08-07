from agent.optimise.runner import (
    _structured_post_synthesis_retry_reason,
    _structured_post_synthesis_retry_suffix,
)


def _verified_record(verdict: str) -> dict:
    return {
        "verdict": verdict,
        "reason": "Fully verified candidate produced no useful objective gain.",
        "synthesis": True,
        "fully_verified": True,
        "performance_comparison": {
            "latency_delta_percent": 0.0,
            "throughput_delta_percent": 0.0,
        },
        "deltas_percent": {
            "resources_lut_used": 0.0,
            "resources_ff_used": 0.0,
            "resources_dsp_used": 0.0,
            "resources_bram_used": 0.0,
        },
    }


def test_verified_synthesis_equivalent_candidate_gets_retry():
    assert (
        _structured_post_synthesis_retry_reason(
            _verified_record("reject_no_change")
        )
        == "reject_no_change"
    )


def test_verified_no_objective_gain_candidate_gets_retry():
    assert (
        _structured_post_synthesis_retry_reason(
            _verified_record("reject_no_objective_gain")
        )
        == "reject_no_objective_gain"
    )


def test_static_no_change_does_not_get_post_synthesis_retry():
    record = _verified_record("reject_no_change")
    record["synthesis"] = None
    record["fully_verified"] = False

    assert _structured_post_synthesis_retry_reason(record) is None


def test_failed_candidate_does_not_get_post_synthesis_retry():
    record = _verified_record("reject_csim")

    assert _structured_post_synthesis_retry_reason(record) is None


def test_feedback_preserves_strategy_and_uses_measured_evidence():
    record = _verified_record("reject_no_change")
    strategy = {"name": "memory_parallelism"}

    prompt = _structured_post_synthesis_retry_suffix(strategy, record)

    assert "Strategy family remains: memory_parallelism" in prompt
    assert "latency: +0.00%" in prompt
    assert "throughput period: +0.00%" in prompt
    assert "serial buffering" in prompt
    assert "same sequential access pattern" in prompt

    # Core logic must remain benchmark-independent.
    lowered = prompt.lower()
    assert "bicg" not in lowered
    assert "atax" not in lowered
    assert "factor=2" not in lowered
    assert "a_row" not in lowered
