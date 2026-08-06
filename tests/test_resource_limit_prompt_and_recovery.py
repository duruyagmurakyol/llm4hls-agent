from __future__ import annotations

from pathlib import Path

from agent.optimise.generate import resource_limit_prompt_suffix
from agent.optimise.parent_selection import select_refinement_parent


def _metrics(latency: float) -> dict[str, float]:
    return {
        "latency_ns": latency,
        "throughput_period_ns": latency,
        "resources_lut_used": 1781.0,
        "resources_ff_used": 1635.0,
        "resources_dsp_used": 26.0,
        "resources_bram_used": 0.0,
    }


def test_every_model_prompt_receives_explicit_resource_ceilings() -> None:
    suffix = resource_limit_prompt_suffix(
        {
            "resource_limits": {
                "lut": 7124,
                "ff": 6540,
                "dsp": 104,
                "bram": 64,
            }
        }
    )

    assert "Hard resource ceilings (mandatory)" in suffix
    assert "LUT <= 7124" in suffix
    assert "FF <= 6540" in suffix
    assert "DSP <= 104" in suffix
    assert "BRAM <= 64" in suffix
    assert "exceeding any ceiling is ineligible" in suffix


def test_no_change_candidate_can_anchor_resource_limit_recovery(
    tmp_path: Path,
) -> None:
    no_change = {
        "candidate_index": 1,
        "candidate_file": str(tmp_path / "candidate_001.cpp"),
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {
            "configured": True,
            "passed": True,
            "violations": [],
        },
        "verdict": "reject_no_change",
        "metrics": _metrics(112819.882),
        "cost": {},
    }
    over_budget = {
        "candidate_index": 2,
        "candidate_file": str(tmp_path / "candidate_002.cpp"),
        "fully_verified": False,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {
            "configured": True,
            "passed": False,
            "violations": [
                {
                    "metric": "resources_ff_used",
                    "limit": 6540.0,
                    "actual": 8701.0,
                    "excess": 2161.0,
                }
            ],
        },
        "verdict": "reject_resource_limits",
        "metrics": {
            **_metrics(68992.626),
            "resources_lut_used": 4511.0,
            "resources_ff_used": 8701.0,
            "resources_dsp_used": 28.0,
        },
        "cost": {},
    }

    selected = select_refinement_parent([no_change, over_budget])

    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 1
    assert reason == "resource_limit_recovery_from_feasible_verified"
