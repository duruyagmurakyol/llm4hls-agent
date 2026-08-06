from __future__ import annotations

import importlib
import json
from pathlib import Path

from agent.optimise.eligibility import annotate_candidate_eligibility
from agent.optimise.parent_selection import (
    BASELINE_RESTART_REASON,
    select_refinement_parent,
)


def _metrics(
    *,
    latency_ns: float,
    lut: int,
    ff: int,
    dsp: int,
    bram: int = 0,
) -> dict:
    return {
        "latency_ns": latency_ns,
        "throughput_period_ns": latency_ns + 1.0,
        "resources_lut_used": lut,
        "resources_ff_used": ff,
        "resources_dsp_used": dsp,
        "resources_bram_used": bram,
    }


def _verified_candidate(
    output_dir: Path,
    index: int,
    *,
    verdict: str,
    latency_ns: float,
    latency_delta: float,
    lut: int,
    lut_delta: float,
    ff: int,
    ff_delta: float,
    dsp: int,
    dsp_delta: float,
) -> dict:
    record = {
        "candidate_index": index,
        "candidate_file": str(output_dir / f"candidate_{index:03d}.cpp"),
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim_required": True,
        "cosim": True,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "meets_resource_limits": True,
        "resource_limit_compliance": {
            "configured": True,
            "passed": True,
            "limits": {
                "resources_lut_used": 7124.0,
                "resources_ff_used": 6540.0,
                "resources_dsp_used": 104.0,
                "resources_bram_used": 64.0,
            },
            "usage": {
                "resources_lut_used": lut,
                "resources_ff_used": ff,
                "resources_dsp_used": dsp,
                "resources_bram_used": 0,
            },
            "violations": [],
        },
        "metrics": _metrics(
            latency_ns=latency_ns,
            lut=lut,
            ff=ff,
            dsp=dsp,
        ),
        "deltas_percent": {
            "latency_ns": latency_delta,
            "throughput_period_ns": latency_delta,
            "resources_lut_used": lut_delta,
            "resources_ff_used": ff_delta,
            "resources_dsp_used": dsp_delta,
            "resources_bram_used": 0.0,
        },
        "cost": {
            "input_tokens": 500,
            "output_tokens": 250,
            "total_tokens": 750,
            "tool_calls": 2,
            "tool_seconds": 10.0,
        },
        "verdict": verdict,
    }
    return annotate_candidate_eligibility(record)


def _rejected_candidate(
    output_dir: Path,
    index: int,
    *,
    verdict: str,
) -> dict:
    record = {
        "candidate_index": index,
        "candidate_file": str(output_dir / f"candidate_{index:03d}.cpp"),
        "static_validation": False,
        "csim": None,
        "synthesis": None,
        "cosim_required": True,
        "cosim": None,
        "fully_verified": False,
        "meets_frequency_requirement": None,
        "meets_resource_limits": None,
        "resource_limit_compliance": {
            "configured": True,
            "passed": False,
            "limits": {},
            "usage": {},
            "violations": [],
        },
        "metrics": {},
        "deltas_percent": {},
        "cost": {
            "input_tokens": 500,
            "output_tokens": 250,
            "total_tokens": 750,
            "tool_calls": 0,
            "tool_seconds": 0.0,
        },
        "verdict": verdict,
    }
    return annotate_candidate_eligibility(record)


def _poor_dsp_only_pareto(output_dir: Path, index: int = 2) -> dict:
    """Reproduce the practically poor Pareto point from the GEMM trace."""
    return _verified_candidate(
        output_dir,
        index,
        verdict="keep_pareto_candidate",
        latency_ns=304784.802,
        latency_delta=170.15,
        lut=2028,
        lut_delta=13.87,
        ff=4624,
        ff_delta=182.81,
        dsp=14,
        dsp_delta=-46.15,
    )


def _assert_baseline_restart(selected: tuple[dict, str] | None) -> None:
    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 0
    assert reason == BASELINE_RESTART_REASON


def test_extreme_dsp_only_pareto_does_not_become_refinement_parent(
    tmp_path: Path,
) -> None:
    """Archive-worthy must not automatically mean refinement-worthy."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    selected = select_refinement_parent([_poor_dsp_only_pareto(output_dir)])

    _assert_baseline_restart(selected)


def test_failed_recovery_branch_restarts_from_baseline(tmp_path: Path) -> None:
    """A failed strategy branch must be retired instead of selected again."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    pareto = _poor_dsp_only_pareto(output_dir)
    failed = _rejected_candidate(output_dir, 3, verdict="reject_static")

    (output_dir / "candidate_003_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_latency_tradeoff",
                "parameters": {"factor": 2},
                "source_candidate_index": 2,
                "next_candidate_index": 3,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "candidate_003_static_validation.json").write_text(
        json.dumps(
            {
                "passed": False,
                "strategy_compliance": {
                    "required": True,
                    "passed": False,
                },
            }
        ),
        encoding="utf-8",
    )

    selected = select_refinement_parent([pareto, failed])

    _assert_baseline_restart(selected)


def test_duplicate_after_failed_recovery_does_not_reselect_same_parent(
    tmp_path: Path,
) -> None:
    """Reproduce the C2 -> C3 static failure -> C4 duplicate dead branch."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    no_gain = _verified_candidate(
        output_dir,
        1,
        verdict="reject_no_objective_gain",
        latency_ns=118883.044,
        latency_delta=5.38,
        lut=2323,
        lut_delta=30.43,
        ff=2210,
        ff_delta=35.17,
        dsp=28,
        dsp_delta=7.69,
    )
    pareto = _poor_dsp_only_pareto(output_dir)
    failed = _rejected_candidate(output_dir, 3, verdict="reject_static")
    duplicate = _rejected_candidate(output_dir, 4, verdict="reject_duplicate")

    (output_dir / "candidate_003_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_latency_tradeoff",
                "parameters": {"factor": 2},
                "source_candidate_index": 2,
                "next_candidate_index": 3,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "candidate_003_static_validation.json").write_text(
        json.dumps(
            {
                "passed": False,
                "strategy_compliance": {
                    "required": True,
                    "passed": False,
                },
            }
        ),
        encoding="utf-8",
    )

    selected = select_refinement_parent(
        [no_gain, pareto, failed, duplicate]
    )

    _assert_baseline_restart(selected)


def test_material_latency_improvement_remains_a_valid_parent(
    tmp_path: Path,
) -> None:
    """The rescue policy must not discard a genuinely useful candidate."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    useful = _verified_candidate(
        output_dir,
        1,
        verdict="accept_dominates_baseline",
        latency_ns=80000.0,
        latency_delta=-29.09,
        lut=1700,
        lut_delta=-4.55,
        ff=1600,
        ff_delta=-2.14,
        dsp=26,
        dsp_delta=0.0,
    )

    selected = select_refinement_parent([useful])

    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 1
    assert reason == "dominates_baseline_candidate"


def test_no_gain_candidate_still_restarts_from_baseline(tmp_path: Path) -> None:
    """Preserve the already-correct independent baseline restart behaviour."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    no_gain = _verified_candidate(
        output_dir,
        1,
        verdict="reject_no_objective_gain",
        latency_ns=118883.044,
        latency_delta=5.38,
        lut=2323,
        lut_delta=30.43,
        ff=2210,
        ff_delta=35.17,
        dsp=28,
        dsp_delta=7.69,
    )

    selected = select_refinement_parent([no_gain])

    _assert_baseline_restart(selected)


def test_five_slot_schedule_starts_with_three_distinct_baseline_explorations() -> None:
    """Define the structured schedule contract before implementing it."""
    search_policy = importlib.import_module("agent.optimise.search_policy")

    schedule = search_policy.build_structured_search_schedule(
        max_candidates=5
    )

    assert len(schedule) == 5
    first_three = schedule[:3]
    assert all(attempt["phase"] == "explore" for attempt in first_three)
    assert all(
        attempt["parent_candidate_index"] == 0
        for attempt in first_three
    )
    assert len(
        {attempt["strategy_family"] for attempt in first_three}
    ) == 3
