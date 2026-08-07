from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise import generate


SOURCE = """#include <stdint.h>
void kernel(int a[8], int b[8]) {
    for (int i = 0; i < 8; ++i) {
        b[i] = a[i] + 1;
    }
}
"""


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "top_function": "kernel",
                "output_dir": "run",
                "model": {
                    "provider": "siliconflow",
                    "name": "Qwen/Qwen3.5-122B-A10B",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "enable_thinking": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_corrective_retry_provider_failure_preserves_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generate, "REPO_ROOT", tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    candidate = output_dir / "candidate_001.cpp"
    candidate.write_text(SOURCE, encoding="utf-8")
    (output_dir / "candidate_001_prompt.txt").write_text(
        "Optimise this source.\n\n"
        + generate.CORRECTIVE_RETRY_MARKER
        + "\nDo not return the parent unchanged.\n",
        encoding="utf-8",
    )
    (output_dir / "candidate_001_model_metadata.json").write_text(
        json.dumps({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}),
        encoding="utf-8",
    )

    def fail_complete(**_: object) -> object:
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(generate, "complete", fail_complete)

    result = generate.generate_candidate(_config(tmp_path), 1)

    assert result == candidate
    assert candidate.read_text(encoding="utf-8") == SOURCE
    failure = json.loads(
        (output_dir / "candidate_001_generation_retry_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["generation_retry_failed"] is True
    assert failure["error_type"] == "RuntimeError"
    assert "simulated provider failure" in failure["error"]
    assert failure["total_tokens"] == 0


def test_initial_provider_failure_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generate, "REPO_ROOT", tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "candidate_001_prompt.txt").write_text(
        "Optimise this source without a corrective retry marker.\n",
        encoding="utf-8",
    )

    def fail_complete(**_: object) -> object:
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(generate, "complete", fail_complete)

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        generate.generate_candidate(_config(tmp_path), 1)
