from __future__ import annotations

import json

from agent.optimise.pareto_frontier import annotate_pareto_frontier


def _metrics(latency: float, throughput: float, lut: int, ff: int) -> dict[str, object]:
    return {
        "latency_ns": latency,
        "throughput_period_ns": throughput,
        "resources_lut_used": lut,
        "resources_ff_used": ff,
        "resources_dsp_used": 0,
        "resources_bram_used": 0,
    }


def test_materialises_current_frontier_and_marks_historical_dominance(tmp_path) -> None:
    candidate_1 = {
        "candidate_index": 1,
        "fully_verified": True,
        "verdict": "keep_pareto_candidate",
        "reason": "trade-off",
        "metrics": _metrics(8.128, 8.128, 441, 10),
    }
    candidate_4 = {
        "candidate_index": 4,
        "fully_verified": True,
        "verdict": "keep_pareto_candidate",
        "reason": "trade-off",
        "metrics": _metrics(18.073, 13.144, 358, 100),
    }
    candidate_5 = {
        "candidate_index": 5,
        "fully_verified": True,
        "verdict": "keep_pareto_candidate",
        "reason": "trade-off",
        "metrics": _metrics(18.073, 13.144, 237, 90),
    }
    candidate_6 = {
        "candidate_index": 6,
        "fully_verified": True,
        "verdict": "keep_pareto_candidate",
        "reason": "trade-off",
        "metrics": _metrics(16.43, 13.144, 130, 17),
    }
    summary = {
        "candidates": [candidate_1, candidate_4, candidate_5, candidate_6],
        "pareto_archive": [dict(candidate_1), dict(candidate_6)],
    }

    result = annotate_pareto_frontier(tmp_path, summary)

    records = {item["candidate_index"]: item for item in result["candidates"]}
    assert records[1]["pareto"] is True
    assert records[6]["pareto"] is True
    assert records[4]["pareto"] is False
    assert records[4]["dominated_by"] == 6
    assert records[4]["verdict"] == "reject_dominated"
    assert records[5]["dominated_by"] == 6
    assert records[5]["verdict"] == "reject_dominated"

    persisted = json.loads((tmp_path / "pareto_frontier.json").read_text())
    assert [item["candidate_index"] for item in persisted["members"]] == [1, 6]
    assert persisted["dominated_candidates"] == [
        {"candidate_index": 4, "dominated_by": 6},
        {"candidate_index": 5, "dominated_by": 6},
    ]
