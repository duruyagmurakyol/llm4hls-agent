from __future__ import annotations

from pathlib import Path

from agent.tools.synthesis import REPO_ROOT, _tcl_parts


def test_vector_add_candidate_keeps_source_include_path(tmp_path: Path) -> None:
    task_cfg = (
        REPO_ROOT
        / "benchmarks/vector_add/faults/functional_subtraction/task.cfg"
    )
    candidate = tmp_path / "candidate.cpp"
    candidate.write_text(
        '#include "vector_add.h"\nvoid vector_add(const int*, const int*, int*) {}\n',
        encoding="utf-8",
    )

    _, design, auxiliaries, top = _tcl_parts(
        task_cfg,
        candidate,
        include_testbench=True,
    )

    include = (task_cfg.parent / "src").resolve().as_posix()
    assert top == "vector_add"
    assert f"-I{include}" in design
    assert candidate.resolve().as_posix() in design

    testbench = next(command for command in auxiliaries if "-tb" in command)
    assert f"-I{include}" in testbench
