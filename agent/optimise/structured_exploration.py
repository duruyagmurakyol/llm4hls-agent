"""Prepare distinct baseline-rooted prompts for structured exploration slots.

The baseline diagnosis pipeline already creates ``candidate_001_prompt.txt``.
This module preserves that diagnosis prompt as an immutable template and adds a
single explicit strategy-family contract for each of the first three search
slots.  It does not call a model or run validation tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.optimise.search_policy import (
    EXPLORATION_STRATEGY_FAMILIES,
    build_structured_search_schedule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PROMPT_TEMPLATE = "baseline_optimisation_prompt.txt"

STRATEGY_GUIDANCE: dict[str, dict[str, Any]] = {
    "critical_path_restructuring": {
        "objective": (
            "Shorten the measured critical dependency path, especially loop-carried "
            "accumulator or reduction chains, without broad resource replication."
        ),
        "required_changes": [
            "Make one focused source-level restructuring in the diagnosed target region.",
            "Use independent partial accumulators, a balanced reduction, or equivalent dependency-shortening structure only when algorithmically valid.",
            "Preserve the exact top-function interface, loop bounds, input coverage, and numerical behaviour.",
        ],
        "forbidden_changes": [
            "Do not use complete loop unrolling.",
            "Do not completely partition top-level interface arrays.",
            "Do not add unrelated DATAFLOW or broad pragma combinations.",
        ],
    },
    "bounded_unroll": {
        "objective": (
            "Increase controlled parallelism with a small bounded unroll factor while "
            "keeping LUT, FF, DSP, BRAM and timing within the configured limits."
        ),
        "required_changes": [
            "Choose one bounded factor from 2 or 4 that is compatible with the loop bounds.",
            "When a reduction is present, use matching independent partial accumulators or an equivalent safe reduction structure.",
            "Keep remainder handling and every input element correct when the trip count is not divisible by the chosen factor.",
        ],
        "forbidden_changes": [
            "Do not completely unroll the target loop.",
            "Do not combine several unrelated optimisation families in this candidate.",
            "Do not completely partition top-level interface arrays.",
        ],
        "parameters": {"allowed_factors": [2, 4]},
    },
    "memory_parallelism": {
        "objective": (
            "Remove a measured or source-supported memory access bottleneck using local "
            "buffering or bounded banking matched to the access pattern."
        ),
        "required_changes": [
            "Apply local buffering, cyclic partitioning, block partitioning, or access reordering only where the diagnosed target requires additional ports.",
            "Match any banking factor to the actual parallel access pattern and keep it bounded.",
            "Preserve all reads, writes, bounds, interfaces and algorithmic behaviour.",
        ],
        "forbidden_changes": [
            "Do not completely partition top-level interface arrays.",
            "Do not copy entire large arrays into local storage without a demonstrated access benefit.",
            "Do not add unbounded replication or unrelated accumulator rewrites.",
        ],
    },
}


def _load_config(config_source: Any) -> dict[str, Any]:
    resolved = config_source.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Optimisation config must contain a JSON object")
    return value


def _output_dir(config: dict[str, Any]) -> Path:
    path = Path(str(config["output_dir"]))
    return path if path.is_absolute() else REPO_ROOT / path


def _strategy_section(candidate_index: int, strategy_family: str) -> str:
    guidance = STRATEGY_GUIDANCE[strategy_family]
    required = "\n".join(
        f"- {item}" for item in guidance["required_changes"]
    )
    forbidden = "\n".join(
        f"- {item}" for item in guidance["forbidden_changes"]
    )
    return f"""

Structured exploration contract:
- Search slot: candidate {candidate_index:03d} of the structured exploration phase.
- Implementation parent: original verified baseline (candidate 000).
- Strategy family: {strategy_family}.
- This candidate must remain independent from the other exploration families.
- Apply only this primary strategy family; measured results will decide later exploitation.

Strategy objective:
{guidance['objective']}

Required changes:
{required}

Forbidden changes:
{forbidden}

Return the complete modified C++ source only. Do not include explanations or Markdown fences.
"""


def prepare_structured_exploration_prompt(
    config_source: Any,
    *,
    candidate_index: int,
    strategy_family: str,
) -> Path:
    """Write one baseline-rooted exploration prompt and its audit metadata."""

    schedule = build_structured_search_schedule(max_candidates=3)
    expected = next(
        (
            item
            for item in schedule
            if item["candidate_index"] == candidate_index
        ),
        None,
    )
    if expected is None or expected["phase"] != "explore":
        raise ValueError("candidate_index is not a structured exploration slot")
    if strategy_family != expected["strategy_family"]:
        raise ValueError(
            f"candidate {candidate_index:03d} requires strategy family "
            f"{expected['strategy_family']}"
        )
    if strategy_family not in EXPLORATION_STRATEGY_FAMILIES:
        raise ValueError(f"unsupported exploration strategy: {strategy_family}")

    config = _load_config(config_source)
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_prompt = output_dir / "candidate_001_prompt.txt"
    template_path = output_dir / BASELINE_PROMPT_TEMPLATE
    if not template_path.is_file():
        if not initial_prompt.is_file():
            raise FileNotFoundError(
                f"Baseline diagnosis prompt not found: {initial_prompt}"
            )
        template_path.write_text(
            initial_prompt.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    base_prompt = template_path.read_text(encoding="utf-8").rstrip()
    prompt = base_prompt + _strategy_section(candidate_index, strategy_family)
    prompt_path = output_dir / f"candidate_{candidate_index:03d}_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    guidance = STRATEGY_GUIDANCE[strategy_family]
    strategy_payload = {
        "name": strategy_family,
        "parameters": dict(guidance.get("parameters") or {}),
        "reason": guidance["objective"],
        "required_changes": list(guidance["required_changes"]),
        "forbidden_changes": list(guidance["forbidden_changes"]),
        "source_candidate_index": 0,
        "next_candidate_index": candidate_index,
        "trigger": "structured_baseline_exploration",
        "phase": "explore",
        "schedule_slot": candidate_index,
    }
    (output_dir / f"candidate_{candidate_index:03d}_strategy.json").write_text(
        json.dumps(strategy_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    feedback_payload = {
        "previous_candidate_index": 0,
        "next_candidate_index": candidate_index,
        "selected_parent": "verified_baseline",
        "strategy_family": strategy_family,
        "phase": "explore",
        "structured_schedule": True,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "template_file": str(template_path.relative_to(REPO_ROOT)),
    }
    (output_dir / f"candidate_{candidate_index:03d}_feedback.json").write_text(
        json.dumps(feedback_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompt_path
