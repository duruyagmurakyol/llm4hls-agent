import json
from pathlib import Path
from types import SimpleNamespace

from agent.terminal_reporting import (
    build_run_summary,
    render_run_terminal,
    render_suite_terminal,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_single_run_summary_uses_authoritative_budget_and_selected_metrics(tmp_path):
    repo = tmp_path / "repo"
    output = repo / "runs" / "task"
    write_json(
        output / "budget_summary.json",
        {
            "consumed": {
                "model_calls": 8,
                "input_tokens": 4485,
                "output_tokens": 934,
                "total_tokens": 5419,
                "csim_calls": 8,
                "synthesis_calls": 7,
                "cosim_calls": 4,
            }
        },
    )
    write_json(
        output / "experiment_summary.json",
        {
            "baseline_metrics": {
                "latency_ns": 125.4,
                "throughput_period_ns": 18.7,
                "resources_lut_used": 96,
                "resources_ff_used": 40,
                "resources_dsp_used": 3,
                "frequency_mhz": 120.0,
            },
            "candidates": [
                {"candidate_index": 1, "fully_verified": False},
                {
                    "candidate_index": 2,
                    "fully_verified": True,
                    "reason": "Fully verified candidate offers a latency-resource trade-off.",
                },
            ],
            "selected_design": {
                "candidate_index": 2,
                "verdict": "keep_pareto_candidate",
                "fully_verified": True,
                "meets_frequency_requirement": True,
                "meets_resource_limits": True,
                "metrics": {
                    "latency_ns": 38.1,
                    "throughput_period_ns": 8.0,
                    "resources_lut_used": 315,
                    "resources_ff_used": 239,
                    "resources_dsp_used": 10,
                    "frequency_mhz": 108.0,
                },
            },
        },
    )
    result = SimpleNamespace(
        task_id="dot_product_accumulator_overwrite_repair_full_agent__run__r01",
        success=True,
        status="terminated_budget",
        termination_reason="final_verification_budget_unavailable",
        output_dir="runs/task",
    )

    summary = build_run_summary(result, elapsed_seconds=392, repo_root=repo)
    rendered = render_run_terminal(summary)

    assert summary["task"] == "dot_product/accumulator_overwrite"
    assert summary["model_calls"] == 8
    assert summary["candidate_count"] == 2
    assert "Final agent result" in rendered
    assert "candidate 2" in rendered
    assert "Input tokens:               4,485" in rendered
    assert "Latency:                    125.4 → 38.1 ns" in rendered
    assert "Runtime:                    6 min 32 sec" in rendered
    assert "Final design verified:      yes" in rendered


def test_baseline_fallback_explains_safe_selection(tmp_path):
    repo = tmp_path / "repo"
    output = repo / "runs" / "task"
    write_json(output / "budget_summary.json", {"consumed": {"model_calls": 8}})
    write_json(
        output / "experiment_summary.json",
        {
            "candidates": [
                {"candidate_index": 1, "fully_verified": True},
                {"candidate_index": 2, "fully_verified": False},
            ],
            "selected_design": {
                "candidate_index": 0,
                "verdict": "baseline",
                "fully_verified": True,
                "meets_frequency_requirement": True,
                "meets_resource_limits": True,
                "metrics": {},
            },
        },
    )
    result = SimpleNamespace(
        task_id="gemm_functional_wrong_sign_repair_full_agent",
        success=True,
        status="completed_with_fallback",
        termination_reason="search_completed",
        output_dir="runs/task",
    )

    rendered = render_run_terminal(
        build_run_summary(result, elapsed_seconds=60, repo_root=repo)
    )
    assert "Final selection:            baseline" in rendered
    assert "Baseline retained:          yes" in rendered
    assert "No fully verified candidate ranked above the baseline" in rendered


def test_suite_summary_prioritises_means_then_totals():
    report = {
        "overall": {
            "runs": 36,
            "successful_runs": 36,
            "optimised_candidate_selected": 16,
            "baseline_retained": 20,
            "mean_candidate_count": 5.666667,
            "mean_fully_verified_candidate_count": 3.222222,
            "mean_budget_model_calls_used": 6.666667,
            "mean_input_tokens": 7312.444444,
            "mean_output_tokens": 2328.527778,
            "mean_total_tokens": 9640.972222,
            "mean_budget_csim_calls_used": 5.694444,
            "mean_budget_synthesis_calls_used": 4.666667,
            "mean_budget_cosim_calls_used": 4.222222,
            "mean_elapsed_seconds": 407.443333,
            "total_candidate_count": 204.0,
            "total_fully_verified_candidate_count": 116.0,
            "total_budget_model_calls_used": 240.0,
            "total_input_tokens": 263248.0,
            "total_output_tokens": 83827.0,
            "total_total_tokens": 347075.0,
            "total_budget_csim_calls_used": 205.0,
            "total_budget_synthesis_calls_used": 168.0,
            "total_budget_cosim_calls_used": 152.0,
            "total_elapsed_seconds": 14667.96,
        }
    }

    rendered = render_suite_terminal(report)
    assert rendered.index("Mean per run") < rendered.index("Complete suite totals")
    assert "Successful final designs:   36 / 36" in rendered
    assert "Model calls:                6.67" in rendered
    assert "Candidates evaluated:       204" in rendered
    assert "Total tokens:               347,075" in rendered
    assert "Runtime:                    4 h 4 min 28 sec" in rendered
