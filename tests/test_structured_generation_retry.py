from __future__ import annotations

import json
from pathlib import Path

from agent.optimise import runner


BASELINE = """#include <stdint.h>
void kernel(int a[8], int b[8]) {
    for (int i = 0; i < 8; ++i) {
        b[i] = a[i] + 1;
    }
}
"""


def _write_strategy(output_dir: Path, name: str, **parameters) -> None:
    (output_dir / "candidate_001_strategy.json").write_text(
        json.dumps(
            {
                "name": name,
                "parameters": parameters,
                "compliance_mode": "advisory",
                "source_candidate_index": 0,
                "required_changes": ["Make one material strategy-specific change."],
            }
        ),
        encoding="utf-8",
    )


def test_noop_structured_generation_requests_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (tmp_path / "baseline.cpp").write_text(BASELINE, encoding="utf-8")
    (output_dir / "candidate_001.cpp").write_text(BASELINE, encoding="utf-8")
    _write_strategy(output_dir, "memory_parallelism")

    reason = runner._structured_generation_retry_reason(
        {"baseline": {"source": "baseline.cpp"}},
        output_dir,
        1,
    )

    assert reason == "no_semantic_change"


def test_unrealised_bounded_unroll_requests_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (tmp_path / "baseline.cpp").write_text(BASELINE, encoding="utf-8")
    candidate = BASELINE.replace("a[i] + 1", "a[i] + 2")
    (output_dir / "candidate_001.cpp").write_text(candidate, encoding="utf-8")
    _write_strategy(output_dir, "bounded_unroll", allowed_factors=[2, 4])

    reason = runner._structured_generation_retry_reason(
        {"baseline": {"source": "baseline.cpp"}},
        output_dir,
        1,
    )

    assert reason == "bounded_unroll_not_realised"


def test_realised_bounded_unroll_does_not_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (tmp_path / "baseline.cpp").write_text(BASELINE, encoding="utf-8")
    candidate = BASELINE.replace(
        "    for (int i = 0; i < 8; ++i) {",
        "    for (int i = 0; i < 8; ++i) {\n        #pragma HLS UNROLL factor=2",
    )
    (output_dir / "candidate_001.cpp").write_text(candidate, encoding="utf-8")
    _write_strategy(output_dir, "bounded_unroll", allowed_factors=[2, 4])

    reason = runner._structured_generation_retry_reason(
        {"baseline": {"source": "baseline.cpp"}},
        output_dir,
        1,
    )

    assert reason is None


def test_retry_prompt_forbids_returning_parent_unchanged() -> None:
    text = runner._structured_retry_suffix(
        {
            "name": "memory_parallelism",
            "required_changes": ["Introduce bounded local banking."],
        },
        "no_semantic_change",
    )

    assert "Do not return the implementation parent unchanged" in text
    assert "bounded local row/tile buffer" in text
    assert "memory_parallelism" in text
