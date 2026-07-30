#!/usr/bin/env python3

"""Generate one HLS optimisation candidate from an existing PPA prompt."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.providers.siliconflow import complete  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_cpp(text: str) -> str:
    fenced = re.search(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()

    if "#include" not in candidate or "kernel_atax" not in candidate:
        raise ValueError(
            "Model response did not contain a recognisable complete HLS C++ source file. "
            "The raw response has been preserved for inspection."
        )

    return candidate + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call SiliconFlow once using a generated PPA optimisation prompt."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=1,
        help="Candidate number used in output filenames (default: 1)",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Config is missing the 'model' object")
    if model_config.get("provider") != "siliconflow":
        raise ValueError("Only the SiliconFlow provider is supported in this stage")

    output_dir = REPO_ROOT / config["output_dir"]
    prompt_path = output_dir / f"candidate_{args.candidate_index:03d}_prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}\n"
            "Run scripts/run_ppa_optimisation.py first."
        )

    prompt = prompt_path.read_text(encoding="utf-8")
    model_name = str(model_config["name"])

    print("\nSiliconFlow candidate generation")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print("Calling the model once...")

    response = complete(
        model=model_name,
        system_prompt=(
            "You are an FPGA HLS optimisation agent. Follow the supplied constraints "
            "exactly and return only one complete compilable C++ source file."
        ),
        user_prompt=prompt,
        temperature=float(model_config.get("temperature", 0.0)),
        max_tokens=int(model_config.get("max_tokens", 4096)),
        enable_thinking=model_config.get("enable_thinking"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"candidate_{args.candidate_index:03d}_model_response.txt"
    metadata_path = output_dir / f"candidate_{args.candidate_index:03d}_model_metadata.json"
    candidate_path = output_dir / f"candidate_{args.candidate_index:03d}.cpp"

    raw_path.write_text(response.content.rstrip() + "\n", encoding="utf-8")

    metadata = {
        "provider": "siliconflow",
        "model": model_name,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "latency_seconds": response.latency_seconds,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "raw_response_file": str(raw_path.relative_to(REPO_ROOT)),
        "candidate_file": str(candidate_path.relative_to(REPO_ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    try:
        candidate_cpp = extract_cpp(response.content)
    except ValueError:
        print(f"Raw response saved: {raw_path.relative_to(REPO_ROOT)}")
        print(f"Metadata saved: {metadata_path.relative_to(REPO_ROOT)}")
        raise

    candidate_path.write_text(candidate_cpp, encoding="utf-8")

    print("\nCandidate generated")
    print(f"Source: {candidate_path.relative_to(REPO_ROOT)}")
    print(f"Raw response: {raw_path.relative_to(REPO_ROOT)}")
    print(f"Metadata: {metadata_path.relative_to(REPO_ROOT)}")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Total tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_seconds:.2f} seconds")
    print("No Vitis synthesis was run and the baseline source was not modified.")


if __name__ == "__main__":
    main()
