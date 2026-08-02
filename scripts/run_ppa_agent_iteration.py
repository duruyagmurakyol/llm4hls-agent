#!/usr/bin/env python3

"""Advance the HLS PPA agent by one guarded, verdict-driven iteration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(name: str, arguments: list[str], allow_failure: bool = False) -> int:
    print(f"\n=== {name} ===", flush=True)
    completed = subprocess.run([sys.executable, *arguments], cwd=REPO_ROOT, check=False)
    if completed.returncode != 0 and not allow_failure:
        raise SystemExit(completed.returncode)
    return completed.returncode


def script(name: str, *arguments: object) -> list[str]:
    return [str(SCRIPTS / name), *[str(value) for value in arguments]]


def module(name: str, *arguments: object) -> list[str]:
    return ["-m", name, *[str(value) for value in arguments]]


def candidate_indices(output_dir: Path) -> list[int]:
    pattern = re.compile(r"candidate_(\d{3})\.cpp$")
    return sorted({
        int(match.group(1))
        for path in output_dir.glob("candidate_*.cpp")
        if (match := pattern.match(path.name))
    })


def refresh_summary(config_path: Path) -> dict[str, Any]:
    run_stage("Evaluate current experiment", module("agent.optimise.evaluate", config_path))
    config = load_json(config_path)
    return load_json(REPO_ROOT / config["output_dir"] / "experiment_summary.json")


def record_for(summary: dict[str, Any], index: int) -> dict[str, Any] | None:
    return next((item for item in summary.get("candidates", []) if item.get("candidate_index") == index), None)


def duplicate_gate(config_path: Path, index: int) -> bool:
    return run_stage(
        "Duplicate detection",
        module("agent.optimise.duplicate", config_path, "--candidate-index", index),
        allow_failure=True,
    ) == 0


def synthesise_existing(config_path: Path, output_dir: Path, index: int, summary: dict[str, Any]) -> None:
    budget = summary["budget"]
    used = int(budget["synthesis_calls_used"])
    maximum = int(budget["max_synthesis_calls"])
    if used >= maximum:
        print(f"\nSTOP: synthesis budget exhausted ({used}/{maximum}).")
        return
    if not duplicate_gate(config_path, index):
        print(f"\nSTOP: candidate {index:03d} rejected as a duplicate. No synthesis was run.")
        refresh_summary(config_path)
        return
    run_stage(
        "Vitis synthesis",
        module("agent.tools.synthesis", "synth", config_path, "--candidate-index", index),
    )
    final = refresh_summary(config_path)
    record = record_for(final, index)
    print("\n=== Iteration complete ===")
    if record:
        print(f"Candidate {index:03d}: {record.get('verdict')}")
        print(record.get("reason"))
    final_budget = final.get("budget", {})
    print(f"Synthesis budget: {final_budget.get('synthesis_calls_used')}/{final_budget.get('max_synthesis_calls')} used")
    print(f"Summary: {(output_dir / 'experiment_summary.json').relative_to(REPO_ROOT)}")


def prepare_prompt(config_path: Path, previous_index: int, next_index: int, previous: dict[str, Any]) -> None:
    if previous.get("verdict") in {"keep_pareto_candidate", "accept_dominates_baseline"}:
        run_stage(
            "Prepare Pareto trade-off refinement prompt",
            script("prepare_ppa_tradeoff_refinement.py", config_path, "--source-index", previous_index, "--next-index", next_index),
        )
    else:
        run_stage(
            "Prepare evidence-driven repair prompt",
            script("prepare_ppa_refinement.py", config_path, "--previous-index", previous_index, "--next-index", next_index),
        )


def evaluate_generated_candidate(config_path: Path, output_dir: Path, index: int, *, allow_synthesis: bool) -> None:
    static_rc = run_stage(
        "Static validation",
        module("agent.tools.validation", config_path, "--candidate-index", index),
        allow_failure=True,
    )
    if static_rc != 0:
        print("\nSTOP: candidate rejected by the static gate. No CSim or synthesis was run.")
        refresh_summary(config_path)
        return
    if not duplicate_gate(config_path, index):
        print("\nSTOP: candidate rejected as a duplicate. No CSim or synthesis was run.")
        refresh_summary(config_path)
        return
    csim_rc = run_stage(
        "Vitis CSim",
        module("agent.tools.synthesis", "csim", config_path, "--candidate-index", index),
        allow_failure=True,
    )
    if csim_rc != 0:
        print("\nSTOP: candidate rejected by CSim. No synthesis was run.")
        refresh_summary(config_path)
        return
    if not allow_synthesis:
        print("\nSTOP: candidate passed all pre-synthesis gates.")
        print("Rerun with --allow-synthesis; the controller will resume this candidate.")
        refresh_summary(config_path)
        return
    synthesise_existing(config_path, output_dir, index, refresh_summary(config_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, validate, and optionally synthesise one HLS PPA candidate.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--allow-synthesis", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = refresh_summary(config_path)
    indices = candidate_indices(output_dir)
    max_candidates = int(config["budget"]["max_candidates"])

    if not indices:
        prompt = output_dir / "candidate_001_prompt.txt"
        if not prompt.is_file():
            raise FileNotFoundError(f"Initial candidate prompt is missing: {prompt.relative_to(REPO_ROOT)}")
        print("\nAutonomous iteration plan")
        print("Feedback source: baseline diagnosis")
        print("Next candidate: 001")
        budget = summary["budget"]
        print(f"Synthesis budget: {budget.get('synthesis_calls_used')}/{budget.get('max_synthesis_calls')} used")
        run_stage("Generate candidate", script("generate_ppa_candidate.py", config_path, "--candidate-index", 1))
        evaluate_generated_candidate(config_path, output_dir, 1, allow_synthesis=args.allow_synthesis)
        return

    latest_index = indices[-1]
    latest = record_for(summary, latest_index)
    if latest and latest.get("verdict") == "incomplete":
        static = output_dir / f"candidate_{latest_index:03d}_static_validation.json"
        csim = output_dir / f"candidate_{latest_index:03d}_csim_validation.json"
        synthesis = output_dir / f"candidate_{latest_index:03d}_synthesis.json"
        static_ok = static.is_file() and load_json(static).get("passed") is True
        csim_ok = csim.is_file() and load_json(csim).get("passed") is True
        synthesis_done = synthesis.is_file() and load_json(synthesis).get("passed") is True
        if static_ok and csim_ok and not synthesis_done:
            print(f"\nResuming pre-synthesis candidate {latest_index:03d}")
            if not args.allow_synthesis:
                print("\nSTOP: candidate passed all cheap gates and is unique." if duplicate_gate(config_path, latest_index) else "\nSTOP: duplicate candidate rejected before synthesis.")
                return
            synthesise_existing(config_path, output_dir, latest_index, summary)
            return

    next_index = latest_index + 1
    if next_index > max_candidates:
        print(f"\nSTOP: candidate budget exhausted ({max_candidates}).")
        return
    completed = [item for item in summary.get("candidates", []) if item.get("verdict") != "incomplete"]
    if not completed:
        raise RuntimeError("No completed candidate is available for feedback refinement.")
    previous = completed[-1]
    previous_index = int(previous["candidate_index"])
    budget = summary["budget"]
    print("\nAutonomous iteration plan")
    print(f"Feedback source candidate: {previous_index:03d} ({previous.get('verdict')})")
    print(f"Next candidate: {next_index:03d}")
    print(f"Synthesis budget: {budget.get('synthesis_calls_used')}/{budget.get('max_synthesis_calls')} used")
    prepare_prompt(config_path, previous_index, next_index, previous)
    run_stage("Generate candidate", script("generate_ppa_candidate.py", config_path, "--candidate-index", next_index))
    evaluate_generated_candidate(config_path, output_dir, next_index, allow_synthesis=args.allow_synthesis)


if __name__ == "__main__":
    main()
