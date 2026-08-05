import json
from pathlib import Path

from agent.reporting import build_final_report, write_final_report


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_extracts_selected_designs_and_authoritative_budgets(tmp_path):
    repo = tmp_path / "repo"
    suite = repo / "runs" / "overnight_repair" / "example"
    task_a = suite / "tasks" / "task_a"
    task_b = suite / "tasks" / "task_b"
    rows = [
        {
            "benchmark": "hls",
            "case": "eval_gemver_transpose_access_repair_full_agent",
            "repetition": 1,
            "task_id": "hls_eval_gemver_transpose_access_repair_full_agent__example__r01",
            "success": True,
            "elapsed_seconds": 10,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "budget_model_calls_used": 0,
            "unified_result": str(task_a / "unified_agent_result.json"),
        },
        {
            "benchmark": "gemm",
            "case": "wrong_sign",
            "repetition": 2,
            "task_id": "gemm_functional_wrong_sign_repair_full_agent__example__r02",
            "success": True,
            "elapsed_seconds": 20,
            "input_tokens": 200,
            "output_tokens": 40,
            "total_tokens": 240,
            "budget_model_calls_used": 0,
            "unified_result": str(task_b / "unified_agent_result.json"),
        },
    ]
    write_json(suite / "summary.json", {"rows": rows})
    write_json(
        task_a / "budget_summary.json",
        {
            "consumed": {
                "iterations": 7,
                "model_calls": 8,
                "csim_calls": 8,
                "synthesis_calls": 8,
                "cosim_calls": 4,
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
            "stop_reason": "model_calls_budget_exhausted",
        },
    )
    write_json(
        task_b / "budget_summary.json",
        {
            "consumed": {
                "iterations": 1,
                "model_calls": 1,
                "csim_calls": 2,
                "synthesis_calls": 2,
                "cosim_calls": 1,
                "input_tokens": 500,
                "output_tokens": 100,
                "total_tokens": 600,
            }
        },
    )
    write_json(
        task_a / "experiment_summary.json",
        {
            "baseline_metrics": {"latency_ns": 100, "resources_lut_used": 50},
            "candidates": [{"candidate_index": 1, "fully_verified": True}],
            "selected_design": {
                "candidate_index": 1,
                "verdict": "accept_dominates_baseline",
                "fully_verified": True,
                "meets_frequency_requirement": True,
                "meets_resource_limits": True,
                "metrics": {"latency_ns": 80, "resources_lut_used": 40},
            },
        },
    )
    write_json(
        task_b / "experiment_summary.json",
        {
            "baseline_metrics": {"latency_ns": 100, "resources_lut_used": 50},
            "candidates": [],
            "selected_design": {
                "candidate_index": 0,
                "verdict": "baseline",
                "fully_verified": True,
                "meets_frequency_requirement": True,
                "meets_resource_limits": True,
                "metrics": {"latency_ns": 100, "resources_lut_used": 50},
            },
        },
    )

    report = build_final_report(suite, repo_root=repo)
    overall = report["overall"]
    assert overall["runs"] == 2
    assert overall["optimised_candidate_selected"] == 1
    assert overall["baseline_retained"] == 1
    assert overall["total_input_tokens"] == 1500
    assert overall["total_output_tokens"] == 300
    assert overall["total_total_tokens"] == 1800
    assert overall["total_budget_model_calls_used"] == 9
    assert report["rows"][0]["benchmark"] == "gemver"
    assert report["rows"][0]["case"] == "transpose_access"
    assert report["rows"][0]["latency_ns_delta_percent"] == -20.0
    assert report["rows"][0]["resources_lut_used_delta_percent"] == -20.0

    paths = write_final_report(suite, repo_root=repo)
    assert all(path.is_file() for path in paths.values())
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "Optimised candidate selected | 1" in markdown
    assert "Model calls | 9" in markdown
