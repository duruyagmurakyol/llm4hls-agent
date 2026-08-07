from scripts.audit_benchmark_runnability import audit_repository


def test_repository_static_runnability_audit() -> None:
    report = audit_repository(vitis=False)
    failures = "\n".join(
        f"{item['kind']} {item['path']}: {item['status']} — {item['detail']}"
        for item in report["hard_failures"]
    )
    assert report["passed"] is True, failures
    assert report["benchmarks_discovered"] > 0
