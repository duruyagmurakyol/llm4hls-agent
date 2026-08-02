from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.diagnose import prepare_refinement_prompt, prepare_tradeoff_prompt
from agent.optimise.duplicate import check_candidate_duplicate, source_digest
from agent.optimise.evaluate import dominates, evaluate_experiment
from agent.optimise.generate import extract_cpp, generate_candidate
from agent.repair.runner import run_repair
from agent.state import SynthesisMetrics
from agent.tools.synthesis import (
    ensure_baseline_synthesis,
    parse_csynth_xml,
    run_candidate_csim,
    run_candidate_synthesis,
)
from agent.tools.validation import classify_failure, validate_ppa_candidate
from agent.workspace import Workspace


def test_clean_packages_import() -> None:
    assert Workspace(Path("workspace")).resolve("src/kernel.cpp") == Path("workspace/src/kernel.cpp")
    assert classify_failure("FAIL index=0 expected=1 actual=0") == "functional"
    assert callable(run_repair)
    assert callable(validate_ppa_candidate)
    assert callable(run_candidate_csim)
    assert callable(run_candidate_synthesis)
    assert callable(ensure_baseline_synthesis)
    assert callable(parse_csynth_xml)
    assert callable(evaluate_experiment)
    assert callable(check_candidate_duplicate)
    assert callable(generate_candidate)
    assert callable(prepare_refinement_prompt)
    assert callable(prepare_tradeoff_prompt)


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


def test_strategy_library_is_benchmark_independent() -> None:
    path = Path("agent/optimise/strategies.json")
    strategies = json.loads(path.read_text(encoding="utf-8"))
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
