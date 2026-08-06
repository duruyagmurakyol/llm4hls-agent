from pathlib import Path
from types import SimpleNamespace

from agent.onboarding_safe import _benchmark_resource_limits


def test_gemm_uses_balanced_resource_ceiling() -> None:
    benchmark = SimpleNamespace(
        name="gemm",
        root=Path("benchmarks/hls_eval/gemm"),
    )

    assert _benchmark_resource_limits(benchmark) == {
        "lut": 7124,
        "ff": 6540,
        "dsp": 104,
        "bram": 64,
    }


def test_bicg_is_detected_from_parent_directory() -> None:
    benchmark = SimpleNamespace(
        name="golden",
        root=Path("benchmarks/bicg/golden"),
    )

    assert _benchmark_resource_limits(benchmark) == {
        "lut": 34964,
        "ff": 86908,
        "dsp": 224,
        "bram": 64,
    }


def test_unlisted_benchmark_has_no_implicit_ceiling() -> None:
    benchmark = SimpleNamespace(
        name="vector_add",
        root=Path("benchmarks/vector_add"),
    )

    assert _benchmark_resource_limits(benchmark) == {}
