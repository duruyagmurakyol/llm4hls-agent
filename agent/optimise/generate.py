"""Generate HLS PPA candidates from prepared optimisation prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from agent.budget import BudgetState
from agent.optimise.refinement_strategy import (
    apply_strategy_directives,
    select_latency_recovery_factor,
)
from agent.providers.siliconflow import complete

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def configured_top(config: dict[str, Any]) -> str:
    top = config.get("top_function")
    if not isinstance(top, str) or not top.strip():
        raise ValueError("Config is missing a non-empty 'top_function' field")
    return top.strip()


def _attach_latency_recovery_factor(
    output_dir: Path,
    strategy_path: Path,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Persist the next untested factor for bounded latency recovery."""
    if strategy.get("name") != "recover_latency_tradeoff":
        return strategy

    parameters = strategy.get("parameters") or {}
    configured = parameters.get("factor")
    if isinstance(configured, int) and configured > 0:
        return strategy

    completed: list[int] = []
    for path in sorted(output_dir.glob("candidate_*_strategy.json")):
        if path == strategy_path:
            continue
        previous = load_json(path)
        if previous.get("name") != "recover_latency_tradeoff":
            continue
        factor = (previous.get("parameters") or {}).get("factor")
        if isinstance(factor, int) and factor > 0 and factor not in completed:
            completed.append(factor)

    factor = select_latency_recovery_factor(completed)
    if factor is None:
        return strategy

    updated = dict(strategy)
    updated_parameters = dict(parameters)
    updated_parameters["factor"] = factor
    updated["parameters"] = updated_parameters
    strategy_path.write_text(
        json.dumps(updated, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated


def _latency_recovery_prompt_suffix(strategy: dict[str, Any] | None) -> str:
    if not strategy or strategy.get("name") != "recover_latency_tradeoff":
        return ""
    factor = int((strategy.get("parameters") or {}).get("factor", 0))
    if factor <= 0:
        return ""
    return (
        "\n\nDeterministic bounded parameter choice:\n"
        f"- In the selected performance-critical loop, apply #pragma HLS PIPELINE II=1 "
        f"and #pragma HLS UNROLL factor={factor}.\n"
        f"- Use exactly unroll factor {factor}; do not completely unroll the loop.\n"
        "- Keep these directives inside the selected loop body, not at function scope."
    )


def extract_cpp(text: str, required_top: str) -> str:
    fenced = re.search(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else text).strip()
    if "#include" not in candidate or not re.search(rf"\b{re.escape(required_top)}\s*\(", candidate):
        raise ValueError(
            "Model response did not contain a recognisable complete HLS C++ source file "
            f"with top function '{required_top}'. The raw response was preserved."
        )
    return candidate + "\n"


def generate_candidate(
    config_path: Path,
    candidate_index: int = 1,
    *,
    budget: BudgetState | None = None,
) -> Path:
    config = load_json(config_path.resolve())
    required_top = configured_top(config)
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Config is missing the 'model' object")
    if model_config.get("provider") != "siliconflow":
        raise ValueError("Only the SiliconFlow provider is supported in this stage")

    output_dir = REPO_ROOT / config["output_dir"]
    prompt_path = output_dir / f"candidate_{candidate_index:03d}_prompt.txt"
    strategy_path = output_dir / f"candidate_{candidate_index:03d}_strategy.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    strategy = load_json(strategy_path) if strategy_path.is_file() else None
    if strategy:
        strategy = _attach_latency_recovery_factor(
            output_dir,
            strategy_path,
            strategy,
        )

    user_prompt = (
        prompt_path.read_text(encoding="utf-8")
        + _latency_recovery_prompt_suffix(strategy)
    )

    model_name = str(model_config["name"])
    print("\nSiliconFlow candidate generation")
    print(f"Model: {model_name}")
    print(f"Required top: {required_top}")
    print(f"Prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print("Calling the model once...")

    stage = f"candidate_{candidate_index:03d}_generation"
    if budget is not None:
        budget.charge_model_call(stage=stage)

    try:
        response = complete(
            model=model_name,
            system_prompt=(
                "You are an FPGA HLS optimisation agent. Follow the supplied constraints "
                "exactly and return only one complete compilable C++ source file."
            ),
            user_prompt=user_prompt,
            temperature=float(model_config.get("temperature", 0.0)),
            max_tokens=int(model_config.get("max_tokens", 4096)),
            enable_thinking=model_config.get("enable_thinking"),
        )
    except Exception:
        if budget is not None:
            budget.update_last_event(success=False)
        raise

    if budget is not None:
        budget.update_last_event(success=True)
        budget.record_model_tokens(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            stage=stage,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"candidate_{candidate_index:03d}_model_response.txt"
    metadata_path = output_dir / f"candidate_{candidate_index:03d}_model_metadata.json"
    candidate_path = output_dir / f"candidate_{candidate_index:03d}.cpp"

    raw_path.write_text(response.content.rstrip() + "\n", encoding="utf-8")
    candidate_source = extract_cpp(response.content, required_top)
    directives_applied = False
    if strategy and strategy.get("name") == "partial_unroll":
        candidate_source = apply_strategy_directives(candidate_source, strategy)
        directives_applied = True

    metadata_path.write_text(json.dumps({
        "provider": "siliconflow",
        "model": model_name,
        "required_top": required_top,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "latency_seconds": response.latency_seconds,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "raw_response_file": str(raw_path.relative_to(REPO_ROOT)),
        "candidate_file": str(candidate_path.relative_to(REPO_ROOT)),
        "strategy_file": (
            str(strategy_path.relative_to(REPO_ROOT))
            if strategy_path.is_file()
            else None
        ),
        "strategy_directives_applied": directives_applied,
    }, indent=2) + "\n", encoding="utf-8")
    candidate_path.write_text(candidate_source, encoding="utf-8")

    print("\nCandidate generated")
    print(f"Source: {candidate_path.relative_to(REPO_ROOT)}")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Total tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_seconds:.2f} seconds")
    if directives_applied:
        print("Selected strategy directives were applied deterministically.")
    elif _latency_recovery_prompt_suffix(strategy):
        print("Selected latency-recovery factor was supplied to the model.")
    print("No Vitis synthesis was run and the baseline source was not modified.")
    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one HLS PPA candidate.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--candidate-index", type=int, default=1)
    args = parser.parse_args()
    generate_candidate(args.config, args.candidate_index)


if __name__ == "__main__":
    main()
