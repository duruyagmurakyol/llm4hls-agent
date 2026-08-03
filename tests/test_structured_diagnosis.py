from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.repair.diagnose import build_diagnosis, format_diagnosis
from agent.repair.runner import _prompts, _run_repair_once
from agent.state import ValidationResult


def test_syntax_diagnosis_extracts_configured_file_and_line() -> None:
    diagnosis = build_diagnosis(
        stage="csim",
        failure_class="syntax_or_compile",
        evidence=[
            "/home/user/repo/benchmark/src/vector_add.cpp:9:27: "
            "error: expected ';' after expression"
        ],
        editable_files=["src/vector_add.cpp"],
        protected_files=["src/vector_add.h", "testbench/vector_add_test.cpp"],
        top_function="vector_add",
        repair_constraints=["Every output must equal a[i] + b[i]."],
    )

    assert diagnosis["stage"] == "csim"
    assert diagnosis["failure_class"] == "syntax_or_compile"
    assert diagnosis["suspected_files"] == ["src/vector_add.cpp"]
    assert diagnosis["suspected_lines"] == [9]
    assert diagnosis["suspected_locations"] == [
        {"file": "src/vector_add.cpp", "line": 9}
    ]
    assert "does not compile" in diagnosis["summary"]
    assert "Preserve the vector_add function name and signature." in diagnosis[
        "repair_constraints"
    ]
    assert "Modify only: src/vector_add.cpp." in diagnosis["repair_constraints"]
    assert "Every output must equal a[i] + b[i]." in diagnosis[
        "repair_constraints"
    ]


def test_linkage_diagnosis_suspects_editable_source_and_referenced_testbench() -> None:
    diagnosis = build_diagnosis(
        stage="csim",
        failure_class="linkage_or_interface",
        evidence=[
            "ld.lld: error: undefined symbol: vector_add(int const*, int const*, int*)",
            "referenced by vector_add_test.cpp:15 (/tmp/vector_add_test.cpp:15)",
        ],
        editable_files=["src/vector_add.cpp"],
        protected_files=["testbench/vector_add_test.cpp"],
    )

    assert diagnosis["suspected_files"][0] == "src/vector_add.cpp"
    assert "testbench/vector_add_test.cpp" in diagnosis["suspected_files"]
    assert diagnosis["suspected_lines"] == [15]


def test_functional_diagnosis_defaults_to_editable_source_without_inventing_line() -> None:
    diagnosis = build_diagnosis(
        stage="csim",
        failure_class="functional_mismatch",
        evidence=["FAIL index=0 expected=17 actual=-15"],
        editable_files=["src/vector_add.cpp"],
        protected_files=["testbench/vector_add_test.cpp"],
    )

    assert diagnosis["suspected_files"] == ["src/vector_add.cpp"]
    assert diagnosis["suspected_lines"] == []
    assert diagnosis["evidence"] == ["FAIL index=0 expected=17 actual=-15"]


def test_prompt_uses_structured_diagnosis_sections(tmp_path: Path) -> None:
    workspace = tmp_path / "benchmark"
    (workspace / "src").mkdir(parents=True)
    (workspace / "testbench").mkdir()
    (workspace / "src/vector_add.cpp").write_text(
        '#include "vector_add.h"\nvoid vector_add() {}\n', encoding="utf-8"
    )
    (workspace / "src/vector_add.h").write_text(
        "void vector_add();\n", encoding="utf-8"
    )
    (workspace / "testbench/vector_add_test.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    config = {
        "editable_files": ["src/vector_add.cpp"],
        "protected_files": [
            "src/vector_add.h",
            "testbench/vector_add_test.cpp",
        ],
        "context_files": [
            "src/vector_add.h",
            "testbench/vector_add_test.cpp",
        ],
        "repair_constraints": ["Preserve functional behaviour."],
    }
    validation = ValidationResult(
        False,
        "syntax_or_compile",
        1,
        ["src/vector_add.cpp:2:20: error: expected ';'"],
    )
    previous = build_diagnosis(
        stage="csim",
        failure_class="functional_mismatch",
        evidence=["expected=17 actual=-15"],
        editable_files=["src/vector_add.cpp"],
    )

    _, prompt = _prompts(
        config,
        workspace,
        validation,
        {
            "attempt": 1,
            "stage": "csim",
            "failure_class": "functional_mismatch",
            "evidence": previous["evidence"],
            "diagnosis": previous,
        },
    )

    assert "Previous structured diagnosis:" in prompt
    assert "Current structured diagnosis:" in prompt
    assert "Summary:" in prompt
    assert "Suspected files:" in prompt
    assert "Suspected lines: 2" in prompt
    assert "Repair constraints:" in prompt
    assert "Failure evidence:" not in prompt
    assert "expected=17 actual=-15" in prompt


def test_repair_attempt_persists_diagnosis_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "testbench").mkdir()
    (benchmark / "src/kernel.cpp").write_text(
        "void kernel() {}\n", encoding="utf-8"
    )
    (benchmark / "src/kernel.h").write_text(
        "void kernel();\n", encoding="utf-8"
    )
    (benchmark / "testbench/kernel_tb.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    config = {
        "repair_mode": "direct_api",
        "experiment_id": "structured_diagnosis",
        "benchmark_source": str(benchmark),
        "editable_files": ["src/kernel.cpp"],
        "protected_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "context_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "host_validation": {
            "command": [
                "bash",
                "-lc",
                "echo 'src/kernel.cpp:4:7: error: expected expression' >&2; exit 1",
            ],
            "run_command": ["true"],
        },
        "independent_validation": {"enabled": False, "command": ["true"]},
        "model": "model",
    }
    response = SimpleNamespace(
        content="void kernel() {}\n",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_seconds=0.1,
        raw_response={},
    )
    monkeypatch.setattr(
        "agent.repair.runner.generate_repair",
        lambda **kwargs: (response.content, response),
    )
    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)
    run_dir = tmp_path / "run"

    passed, _, result = _run_repair_once(
        config,
        run_dir=run_dir,
        keep_workspace=True,
    )

    assert not passed
    assert (run_dir / "diagnosis_before.json").is_file()
    assert (run_dir / "diagnosis_after.json").is_file()
    assert result["diagnosis"]["failure_class"] == "syntax_or_compile"
    assert result["diagnosis"]["suspected_files"] == ["src/kernel.cpp"]
    assert result["diagnosis"]["suspected_lines"] == [4]
    assert result["final_diagnosis"] == result["feedback"]["diagnosis"]
    assert "Current structured diagnosis:" in (run_dir / "prompt.txt").read_text(
        encoding="utf-8"
    )


def test_diagnosis_formatter_is_compact() -> None:
    diagnosis = build_diagnosis(
        stage="csim",
        failure_class="functional_mismatch",
        evidence=["expected=17 actual=-15"],
        editable_files=["src/vector_add.cpp"],
    )

    rendered = format_diagnosis(diagnosis)

    assert rendered.count("expected=17 actual=-15") == 1
    assert "Stage: csim" in rendered
    assert "Failure class: functional_mismatch" in rendered
