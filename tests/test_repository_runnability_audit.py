from scripts.audit_benchmark_runnability import audit_repository


def test_repository_static_runnability_audit() -> None:
    report = audit_repository(vitis=False)
    assert report["passed"] is True, report["hard_failures"]
    assert report["benchmarks_discovered"] > 0
