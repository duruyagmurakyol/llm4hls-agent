from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from agent.build_feedback import build_feedback
from agent.evaluate_hls import evaluate_candidate
from agent.generate_candidate import write_candidate
from agent.score_candidate import INVALID_SCORE, score_candidate


STRATEGIES = [
    (
        "Focus exclusively on meeting timing. "
        "Restructure floating-point accumulation to reduce the critical path. "
        "Consider multiple independent partial accumulators or a balanced "
        "reduction tree. Merely adding PIPELINE pragmas is not sufficient."
    ),
    (
        "Explore loop unrolling with independent partial sums. "
        "Avoid a single loop-carried floating-point accumulator dependency. "
        "Handle the N=42 array bound correctly."
    ),
    (
        "Explore array partitioning and parallel memory access. "
        "Only use partitioning that enables actual arithmetic parallelism. "
        "Do not increase resources without improving timing."
    ),
    (
        "Try a two-phase ATAX architecture. "
        "First compute all tmp values, then compute y using independent "
        "partial accumulators. Avoid fusing the phases when this creates "
        "long floating-point dependency chains."
    ),
    (
        "Inspect the implementation as a timing optimisation problem. "
        "Do not prioritise LUT reduction unless the estimated clock period "
        "also improves. Reduce the critical path before optimising resources."
    ),
]


def save_json(path: Path, value: Any) -> None:
    """Write a Python value to a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def candidate_name(run_name: str, iteration: int) -> str:
    """Return the generated candidate name for an iteration."""
    return f"atax_agent_{run_name}_{iteration:02d}"


def classify_result(result: dict[str, Any]) -> str:
    """Classify an evaluation result."""
    if result.get("timeout", False):
        return "timeout"

    if not result.get("csim_pass", False):
        return "functional_failure"

    if not result.get("synth_pass", False):
        return "synthesis_failure"

    metrics = result.get("metrics", {})

    estimated_clock = metrics.get("estimated_clock_period_ns")
    target_clock = metrics.get("target_clock_period_ns")

    if estimated_clock is None or target_clock is None:
        return "metrics_failure"

    if float(estimated_clock) > float(target_clock):
        return "timing_failure"

    return "success"


def metric(result: dict[str, Any], key: str) -> Any:
    """Safely retrieve a synthesis metric."""
    return result.get("metrics", {}).get(key)


def print_result(
    *,
    prefix: str,
    candidate: str,
    outcome: str,
    score: float,
    result: dict[str, Any],
) -> None:
    """Print a compact evaluation summary."""
    print(
        f"{prefix} {candidate}: "
        f"outcome={outcome}, "
        f"clock={metric(result, 'estimated_clock_period_ns')}, "
        f"latency={metric(result, 'latency_worst_cycles')}, "
        f"LUT={metric(result, 'resources_lut_used')}, "
        f"DSP={metric(result, 'resources_dsp_used')}, "
        f"score={score}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the closed-loop LLM-based ATAX HLS optimiser."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="SiliconFlow model identifier.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of optimisation iterations.",
    )

    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique name for this optimisation run.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature used for model generation.",
    )

    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("candidates/atax_candidate_3b.cpp"),
        help="Path to the seed ATAX source file.",
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/atax"),
        help="Directory in which optimisation runs are stored.",
    )

    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0.0 and 2.0")

    seed_path: Path = args.seed

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed candidate not found: {seed_path}")

    run_dir = args.runs_dir / args.run_name
    optimisation_dir = run_dir / "candidates"
    seed_evaluation_dir = run_dir / "seed_evaluation"

    run_dir.mkdir(parents=True, exist_ok=True)
    optimisation_dir.mkdir(parents=True, exist_ok=True)

    original_source = seed_path.read_text()
    current_source = original_source

    best_score = INVALID_SCORE
    best_candidate: str | None = None
    best_result: dict[str, Any] | None = None
    best_source = original_source

    history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Evaluate seed candidate
    # ------------------------------------------------------------------

    seed_result = evaluate_candidate(
        candidate_path=str(seed_path),
        benchmark_dir="benchmarks/hls_eval/atax",
        run_dir=str(seed_evaluation_dir),
    )

    seed_outcome = classify_result(seed_result)
    seed_score = score_candidate(seed_result)

    seed_record = {
        "candidate": seed_path.stem,
        "iteration": 0,
        "model": None,
        "strategy": "seed",
        "outcome": seed_outcome,
        "score": seed_score,
        "improved_best": seed_score > best_score,
        "result": seed_result,
    }

    history.append(seed_record)

    print_result(
        prefix="[seed]",
        candidate=seed_path.stem,
        outcome=seed_outcome,
        score=seed_score,
        result=seed_result,
    )

    if seed_score > best_score:
        best_score = seed_score
        best_candidate = seed_path.stem
        best_result = seed_result
        best_source = original_source
        current_source = original_source

    save_json(run_dir / "seed_result.json", seed_result)
    save_json(run_dir / "history.json", history)

    previous_result = seed_result

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------

    for iteration in range(1, args.iterations + 1):
        name = candidate_name(args.run_name, iteration)

        iteration_dir = run_dir / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        candidate_path = iteration_dir / "candidate.cpp"

        strategy = STRATEGIES[(iteration - 1) % len(STRATEGIES)]

        prompt = build_feedback(
            original_source=original_source,
            candidate_source=current_source,
            result=previous_result,
            iteration=iteration,
            strategy=strategy,
            previous_attempts=history[1:],
        )

        prompt_path = iteration_dir / "prompt.txt"
        prompt_path.write_text(prompt)

        try:
            write_candidate(
                prompt=prompt,
                output_path=candidate_path,
                model=args.model,
                temperature=args.temperature,
            )
        except Exception as exc:
            record = {
                "candidate": name,
                "iteration": iteration,
                "model": args.model,
                "strategy": strategy,
                "outcome": "generation_failure",
                "error": str(exc),
                "score": INVALID_SCORE,
                "improved_best": False,
            }

            history.append(record)
            save_json(iteration_dir / "result.json", record)
            save_json(run_dir / "history.json", history)

            print(
                f"[{iteration}/{args.iterations}] {name}: "
                f"outcome=generation_failure, error={exc}"
            )

            # Continue so a temporary model failure does not destroy the run.
            continue

        if not candidate_path.exists():
            record = {
                "candidate": name,
                "iteration": iteration,
                "model": args.model,
                "strategy": strategy,
                "outcome": "generation_failure",
                "error": "The model call completed but no candidate file was created.",
                "score": INVALID_SCORE,
                "improved_best": False,
            }

            history.append(record)
            save_json(iteration_dir / "result.json", record)
            save_json(run_dir / "history.json", history)

            print(
                f"[{iteration}/{args.iterations}] {name}: "
                "outcome=generation_failure, "
                "error=no candidate file was created"
            )

            continue

        generated_source = candidate_path.read_text().strip()

        if not generated_source:
            record = {
                "candidate": name,
                "iteration": iteration,
                "model": args.model,
                "strategy": strategy,
                "outcome": "generation_failure",
                "error": "The generated candidate source file was empty.",
                "score": INVALID_SCORE,
                "improved_best": False,
            }

            history.append(record)
            save_json(iteration_dir / "result.json", record)
            save_json(run_dir / "history.json", history)

            print(
                f"[{iteration}/{args.iterations}] {name}: "
                "outcome=generation_failure, "
                "error=empty candidate source"
            )

            continue

        evaluation_dir = iteration_dir / "evaluation"

        try:
            result = evaluate_candidate(
                candidate_path=str(candidate_path),
                benchmark_dir="benchmarks/hls_eval/atax",
                run_dir=str(evaluation_dir),
            )
        except Exception as exc:
            record = {
                "candidate": name,
                "iteration": iteration,
                "model": args.model,
                "strategy": strategy,
                "outcome": "evaluation_failure",
                "error": str(exc),
                "score": INVALID_SCORE,
                "improved_best": False,
            }

            history.append(record)
            save_json(iteration_dir / "result.json", record)
            save_json(run_dir / "history.json", history)

            print(
                f"[{iteration}/{args.iterations}] {name}: "
                f"outcome=evaluation_failure, error={exc}"
            )

            continue

        outcome = classify_result(result)
        score = score_candidate(result)
        improved_best = score > best_score

        record = {
            "candidate": name,
            "iteration": iteration,
            "model": args.model,
            "strategy": strategy,
            "outcome": outcome,
            "score": score,
            "improved_best": improved_best,
            "result": result,
        }

        history.append(record)

        print_result(
            prefix=f"[{iteration}/{args.iterations}]",
            candidate=name,
            outcome=outcome,
            score=score,
            result=result,
        )

        save_json(iteration_dir / "result.json", record)
        save_json(run_dir / "history.json", history)

        if improved_best:
            best_score = score
            best_candidate = name
            best_result = result
            best_source = generated_source
            current_source = generated_source

            best_candidate_path = run_dir / "best_candidate.cpp"
            shutil.copyfile(candidate_path, best_candidate_path)

            print(f"New best candidate: {name}")
        else:
            current_source = best_source

            print(
                "Candidate did not improve search score: "
                f"{score:.3f} <= {best_score:.3f}. "
                "Continuing from the previous best source."
            )

        # The next feedback prompt should describe the candidate that was
        # just evaluated, even when it was not accepted as the new best.
        previous_result = result

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "temperature": args.temperature,
        "iterations_requested": args.iterations,
        "iterations_completed": len(
            [
                record
                for record in history
                if int(record.get("iteration", 0)) > 0
            ]
        ),
        "best_candidate": best_candidate,
        "best_score": best_score,
        "best_result": best_result,
    }

    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "history.json", history)

    print("\nRun complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()