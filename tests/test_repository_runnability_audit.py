from scripts.audit_benchmark_runnability import (
    _tool_configuration_failure,
    audit_repository,
)


def test_repository_static_runnability_audit() -> None:
    report = audit_repository(vitis=False)
    failures = "\n".join(
        f"{item['kind']} {item['path']}: {item['status']} — {item['detail']}"
        for item in report["hard_failures"]
    )
    assert report["passed"] is True, failures
    assert report["benchmarks_discovered"] > 0


def test_audit_distinguishes_configuration_failure_from_design_failure() -> None:
    config_error = {
        "evidence": [
            "ERROR: [HLS 200-101] add_files: Unknown option '-I/tmp/candidates'."
        ]
    }
    functional_failure = {
        "evidence": [
            "FAIL x[0]: expected 2.0, got 1.0",
            "ERROR: [SIM 211-100] 'csim_design' failed: nonzero return value.",
        ]
    }

    assert _tool_configuration_failure(config_error) is True
    assert _tool_configuration_failure(functional_failure) is False
