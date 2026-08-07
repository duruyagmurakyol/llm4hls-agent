from __future__ import annotations

from pathlib import Path

from agent.baseline import _copy_synthesis_diagnostic_evidence


def test_promoted_baseline_preserves_hls_scheduling_log(tmp_path: Path) -> None:
    project = tmp_path / "synthesis"
    report_dir = project / "solution1/syn/report"
    report_dir.mkdir(parents=True)

    log = project / "vitis_hls.log"
    warning = (
        "WARNING: [HLS 200-448] Lower bound of II is 19 due to multiple "
        "'load' operation 64 bit on array 'A' accessing core:RAM:A\n"
    )
    log.write_text(warning, encoding="utf-8")

    rpt = report_dir / "kernel_bicg_csynth.rpt"
    rpt.write_text("Synthesis report\n", encoding="utf-8")

    stable = tmp_path / "verified_baseline_project"
    copied = _copy_synthesis_diagnostic_evidence(
        {"log_path": str(log)},
        project,
        stable,
    )

    copied_files = [path for path in stable.rglob("*") if path.is_file()]
    assert copied
    assert any(path.name == "vitis_hls.log" for path in copied_files)
    assert any(path.name == "kernel_bicg_csynth.rpt" for path in copied_files)
    assert any("HLS 200-448" in path.read_text(encoding="utf-8") for path in copied_files)
