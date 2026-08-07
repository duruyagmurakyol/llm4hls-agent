from pathlib import Path

from agent.onboarding_safe import resolve_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_atax_auto_discovery_uses_canonical_u55c_source_and_config() -> None:
    task = resolve_benchmark(REPO_ROOT / "benchmarks" / "hls_eval" / "atax")

    assert task.data["artifacts"]["source"] == "benchmarks/hls_eval/atax/src/atax.cpp"
    assert task.data["artifacts"]["build_files"] == ["benchmarks/hls_eval/atax/task.cfg"]
    assert task.data["target"]["part"] == "xcu55c-fsvh2892-2L-e"
    assert task.data["target"]["clock_period_ns"] == 10.0
