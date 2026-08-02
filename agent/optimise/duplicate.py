"""Detect duplicate generated HLS optimisation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalise_source(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"\s+", "", text)


def source_digest(text: str) -> str:
    return hashlib.sha256(normalise_source(text).encode("utf-8")).hexdigest()


def check_candidate_duplicate(config_path: Path, candidate_index: int) -> dict[str, Any]:
    config = load_json(config_path.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    candidate_path = output_dir / f"candidate_{candidate_index:03d}.cpp"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate source not found: {candidate_path}")

    candidate_hash = source_digest(candidate_path.read_text(encoding="utf-8"))
    duplicate_of: int | None = None
    for index in range(1, candidate_index):
        other = output_dir / f"candidate_{index:03d}.cpp"
        if other.is_file() and source_digest(other.read_text(encoding="utf-8")) == candidate_hash:
            duplicate_of = index
            break

    report = {
        "candidate_index": candidate_index,
        "candidate_file": str(candidate_path.relative_to(REPO_ROOT)),
        "normalised_sha256": candidate_hash,
        "is_duplicate": duplicate_of is not None,
        "duplicate_of": duplicate_of,
        "passed": duplicate_of is None,
    }
    report_path = output_dir / f"candidate_{candidate_index:03d}_duplicate_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicate generated HLS candidates.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--candidate-index", type=int, required=True)
    args = parser.parse_args()

    report = check_candidate_duplicate(args.config, args.candidate_index)
    print("\nCandidate duplicate check")
    print(f"Candidate: {report['candidate_file']}")
    if report["passed"]:
        print("Overall: PASS — source is unique among earlier candidates")
    else:
        print(f"Overall: FAIL — duplicates candidate_{report['duplicate_of']:03d}")
    print("No compilation, CSim, or synthesis was run.")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
