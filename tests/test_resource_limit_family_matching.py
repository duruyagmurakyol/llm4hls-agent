from pathlib import Path
from types import SimpleNamespace

from agent.onboarding_safe import BENCHMARK_RESOURCE_LIMITS, _benchmark_resource_limits


def benchmark(name: str) -> SimpleNamespace:
    root = Path("/tmp/smoke_inputs") / name
    return SimpleNamespace(name=name, root=root)


def test_gemm_u55c_inherits_gemm_resource_limits() -> None:
    assert _benchmark_resource_limits(benchmark("gemm_u55c")) == BENCHMARK_RESOURCE_LIMITS["gemm"]


def test_bicg_hyphen_suffix_inherits_bicg_resource_limits() -> None:
    assert _benchmark_resource_limits(benchmark("bicg-u55c")) == BENCHMARK_RESOURCE_LIMITS["bicg"]


def test_unrelated_benchmark_does_not_inherit_limits() -> None:
    assert _benchmark_resource_limits(benchmark("megamm_u55c")) == {}
