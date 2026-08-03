from __future__ import annotations

import json
from pathlib import Path

from agent.config import TaskManifest, load_task
from agent.onboarding import discover_benchmark, onboard_benchmark
from agent.optimise.config_source import InMemoryConfig, as_config_source, ppa_config_from_task
from agent.optimise.diagnose import prepare_refinement_prompt, prepare_tradeoff_prompt
from agent.optimise.duplicate import check_candidate_duplicate, source_digest
from agent.optimise.evaluate import classify_candidate, dominates, evaluate_experiment
from agent.optimise.generate import extract_cpp, generate_candidate
from agent.optimise.runner import OptimisationRunResult, _status_summary, run_optimisation
from agent.repair.runner import run_repair
from agent.state import SynthesisMetrics
from agent.tools.command_runner import CommandResult, run_command
from agent.tools.synthesis import (
    ensure_baseline_synthesis,
    parse_csynth_xml,
    run_candidate_csim,
    run_candidate_synthesis,
    run_csim,
)
from agent.tools.validation import (
    _complete_partition_issues,
    _dataflow_pipeline_conflict,
    _pipeline_complete_unroll_conflicts,
    classify_failure,
    validate_ppa_candidate,
)
from agent.workspace import Workspace


def test_clean_packages_import() -> None:
    assert Workspace(Path("workspace")).resolve("src/kernel.cpp") == Path("workspace/src/kernel.cpp")
    assert classify_failure("FAIL index=0 expected=1 actual=0") == "functional"
    assert callable(run_repair)
    assert callable(validate_ppa_candidate)
    assert callable(run_csim)
    assert callable(run_candidate_csim)
    assert callable(run_candidate_synthesis)
    assert callable(ensure_baseline_synthesis)
    assert callable(parse_csynth_xml)
    assert callable(evaluate_experiment)
    assert callable(check_candidate_duplicate)
    assert callable(generate_candidate)
    assert callable(prepare_refinement_prompt)
    assert callable(prepare_tradeoff_prompt)
    assert callable(run_optimisation)
    assert callable(discover_benchmark)
    assert callable(onboard_benchmark)
    assert OptimisationRunResult.__dataclass_fields__


def test_command_runner_records_execution(tmp_path: Path) -> None:
    result = run_command(
        ["/bin/sh", "-c", "printf ok"],
        cwd=tmp_path,
        echo=False,
        timeout_seconds=2,
    )

    assert result.passed
    assert result.output == "ok"
    assert result.cwd == str(tmp_path.resolve())
    assert result.environment is None
    assert result.timed_out is False
    assert result.return_code == 0
    assert result.exception is None
    assert result.elapsed_seconds >= 0


def test_command_runner_records_timeout(tmp_path: Path) -> None:
    result = run_command(
        ["/bin/sh", "-c", "printf start; sleep 5"],
        cwd=tmp_path,
        echo=False,
        timeout_seconds=0.1,
    )

    assert not result.passed
    assert result.timed_out
    assert "start" in result.output
    assert result.exception is not None
    assert "TimeoutExpired" in result.exception


def test_command_runner_records_start_failure(tmp_path: Path) -> None:
    result = run_command(
        ["missing-llm4hls-command"],
        cwd=tmp_path,
        echo=False,
        timeout_seconds=1,
    )

    assert not result.passed
    assert result.return_code is None
    assert result.exception is not None
    assert "FileNotFoundError" in result.exception


def test_run_csim_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "testbench").mkdir()
    (benchmark / "src" / "kernel.cpp").write_text("void kernel() {}\n", encoding="utf-8")
    (benchmark / "src" / "kernel.h").write_text("void kernel();\n", encoding="utf-8")
    (benchmark / "testbench" / "kernel_tb.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    build_file = benchmark / "task.cfg"
    build_file.write_text(
        "[hls]\n"
        "syn.file=src/kernel.cpp\n"
        "syn.top=kernel\n"
        "tb.file=testbench/kernel_tb.cpp\n"
        "part=xczu3eg-sfvc784-2-e\n"
        "clock=10ns\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.cpp"
    candidate.write_text("void kernel() {}\n", encoding="utf-8")
    task = TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "test_csim",
            "artifacts": {"build_files": [str(build_file)]},
            "output_dir": str(tmp_path / "output"),
        },
    )

    def fake_vitis(tcl: Path, cwd: Path, log_path: Path, timeout_seconds: int) -> CommandResult:
        output = f"Compiling {candidate.name}\n"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        return CommandResult(
            command=("vitis-run", "--mode", "hls", "--tcl", str(tcl)),
            return_code=0,
            output=output,
            cwd=str(cwd),
            timed_out=False,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr("agent.tools.synthesis._run_vitis", fake_vitis)
    report = run_csim(task, candidate)

    assert report["passed"]
    assert report["timed_out"] is False
    assert report["return_code"] == 0
    assert report["failure_class"] == "none"
    assert report["evidence"] == []
    assert report["command"][0] == "vitis-run"
    assert report["duration_seconds"] == 0.1
    assert len(report["candidate_hash"]) == 64
    assert Path(report["log_path"]).is_file()


def test_autonomous_manifests_have_no_secondary_config() -> None:
    for path in (
        Path("configs/tasks/vector_add_track_a.json"),
        Path("configs/tasks/atax_track_a.json"),
    ):
        task = load_task(path)
        assert task.adapter_kind == "autonomous_ppa"
        assert set(task.data["adapter"]) == {"kind"}

        config = ppa_config_from_task(task)
        source = as_config_source(task)
        assert isinstance(source, InMemoryConfig)
        assert config["baseline"]["source"] == task.data["artifacts"]["source"]
        assert config["baseline"]["tcl"] == task.data["artifacts"]["build_files"][0]
        assert config["model"] == task.data["model"]
        assert config["budget"]["max_candidates"] == task.data["budgets"]["max_iterations"]


def test_atax_benchmark_is_discovered_from_tcl() -> None:
    benchmark = discover_benchmark(Path("benchmarks/hls_eval/atax"))
    assert benchmark.name == "atax"
    assert benchmark.top_function == "kernel_atax"
    assert benchmark.part == "xczu3eg-sfvc784-2-e"
    assert benchmark.clock_period_ns == 10.0
    assert benchmark.source.name == "atax_candidate_3b.cpp"
    assert [path.name for path in benchmark.testbenches] == ["atax_tb.cpp"]
    assert benchmark.tcl.name == "run_candidate_3b.tcl"


def test_uninitialised_status_does_not_require_baseline_metrics(tmp_path: Path) -> None:
    config = {
        "experiment_name": "fresh",
        "benchmark": "fresh_benchmark",
        "baseline": {},
        "budget": {"max_candidates": 3, "max_synthesis_calls": 2},
    }
    status, summary = _status_summary(config, tmp_path)
    assert status == "status_uninitialised"
    assert summary["baseline_ready"] is False
    assert summary["candidate_count"] == 0


def test_candidate_extraction_requires_configured_top() -> None:
    source = '#include "kernel.h"\nvoid kernel(int *a) { a[0] = 1; }\n'
    assert extract_cpp(source, "kernel") == source


def test_generic_pareto_dominance() -> None:
    better = SynthesisMetrics(10, 1, 1.0, 20, 30, 1, 0)
    worse = SynthesisMetrics(12, 1, 1.0, 25, 35, 1, 0)
    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_duplicate_digest_ignores_comments_and_whitespace() -> None:
    left = "int f() { return 1; } // comment\n"
    right = "/* note */ int  f(){return 1;}\n"
    assert source_digest(left) == source_digest(right)


def test_complete_partition_guard_targets_interface_arrays() -> None:
    source = """
void vector_add(const float a[1024], float c[1024]) {
#pragma HLS ARRAY_PARTITION variable=a complete
#pragma HLS ARRAY_PARTITION variable=local complete
float local[4];
}
"""
    issues = _complete_partition_issues(source, "vector_add")
    assert [item["variable"] for item in issues] == ["a"]


def test_dataflow_pipeline_conflict_guard() -> None:
    source = """
void vector_add(const float a[16], float c[16]) {
#pragma HLS DATAFLOW
#pragma HLS PIPELINE II=1
for (int i = 0; i < 16; ++i) c[i] = a[i];
}
"""
    assert _dataflow_pipeline_conflict(source, "vector_add")


def test_pipeline_complete_unroll_conflict_guard() -> None:
    source = """
void kernel(double a[42]) {
dot_loop:
for (int j = 0; j < 42; ++j) {
#pragma HLS PIPELINE
#pragma HLS UNROLL
    a[j] += 1.0;
}
}
"""
    conflicts = _pipeline_complete_unroll_conflicts(source, "kernel")
    assert conflicts
    assert conflicts[0]["loop_label"] == "dot_loop"


def test_partial_unroll_can_coexist_with_pipeline() -> None:
    source = """
void kernel(double a[42]) {
dot_loop:
for (int j = 0; j < 42; ++j) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=2
    a[j] += 1.0;
}
}
"""
    assert not _pipeline_complete_unroll_conflicts(source, "kernel")


def test_timeout_is_terminal_candidate_verdict(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "candidate_001.cpp"
    candidate.write_text("void kernel() {}\n", encoding="utf-8")
    (tmp_path / "candidate_001_static_validation.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (tmp_path / "candidate_001_csim_validation.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (tmp_path / "candidate_001_synthesis.json").write_text(
        json.dumps({"passed": False, "synthesis_run": True, "timed_out": True, "timeout_seconds": 600}),
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path.parent)
    record = classify_candidate(tmp_path, 1, {}, {})
    assert record["verdict"] == "reject_synthesis_timeout"
    assert record["synthesis_run"] is True


def test_strategy_library_is_benchmark_independent() -> None:
    strategies = json.loads(Path("agent/optimise/strategies.json").read_text(encoding="utf-8"))
    assert strategies
    text = json.dumps(strategies).lower()
    assert "vector_add" not in text
    assert "atax" not in text
    assert "bicg" not in text


def test_obsolete_scripts_are_removed() -> None:
    obsolete = [
        "scripts/run_api_experiment.py",
        "scripts/run_experiment.py",
        "scripts/run_structured_experiment.py",
        "scripts/run_suite.py",
        "scripts/validate_ppa_candidate.py",
        "scripts/run_ppa_candidate_csim.py",
        "scripts/run_ppa_candidate_synthesis.py",
        "scripts/ensure_ppa_baseline_synthesis.py",
        "scripts/evaluate_ppa_experiment.py",
        "scripts/detect_ppa_candidate_duplicate.py",
        "scripts/generate_ppa_candidate.py",
        "scripts/prepare_ppa_refinement.py",
        "scripts/prepare_ppa_tradeoff_refinement.py",
    ]
    assert all(not Path(path).exists() for path in obsolete)


def test_bicg_benchmark_is_discovered_from_task_cfg() -> None:
    benchmark = discover_benchmark(Path("benchmarks/bicg/golden"))

    assert benchmark.name == "bicg"
    assert benchmark.top_function == "kernel_bicg"
    assert benchmark.part == "xczu3eg-sfvc784-2-e"
    assert benchmark.clock_period_ns == 10.0
    assert benchmark.source.name == "bicg.cpp"
    assert [path.name for path in benchmark.testbenches] == ["bicg_test.cpp"]
    assert benchmark.tcl.name == "task.cfg"
