from __future__ import annotations

import hashlib
import json

import pytest

from agent.resume import load_resumable_baseline


def _write_baseline(tmp_path, *, verified: bool = True, matching_hash: bool = True):
    output = tmp_path / "out"
    output.mkdir()
    source = output / "active_baseline.cpp"
    source.write_text("void kernel() {}\n", encoding="utf-8")
    project = output / "verified_baseline_project"
    report = project / "solution1/syn/report/kernel_csynth.xml"
    report.parent.mkdir(parents=True)
    report.write_text("<Report/>\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {
        "source": str(source),
        "candidate_hash": digest if matching_hash else "0" * 64,
        "project_dir": str(project),
        "top_csynth_xml": str(report),
        "metrics": {"latency_best_cycles": 8},
        "validation": {
            "csim_passed": verified,
            "synthesis_passed": verified,
            "cosim_passed": verified,
        },
    }
    (output / "verified_baseline.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    return output, record


def test_loads_fully_verified_matching_baseline(tmp_path) -> None:
    output, expected = _write_baseline(tmp_path)

    assert load_resumable_baseline(output) == expected


def test_rejects_unverified_baseline(tmp_path) -> None:
    output, _ = _write_baseline(tmp_path, verified=False)

    with pytest.raises(ValueError, match="not fully verified"):
        load_resumable_baseline(output)


def test_rejects_hash_mismatch(tmp_path) -> None:
    output, _ = _write_baseline(tmp_path, matching_hash=False)

    with pytest.raises(ValueError, match="hash"):
        load_resumable_baseline(output)
