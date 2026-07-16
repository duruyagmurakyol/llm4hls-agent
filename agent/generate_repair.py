#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a generated HLS repair prompt to Codex."
    )
    parser.add_argument(
        "prompt",
        type=Path,
        help="Path to repair_prompt.txt",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Codex model to use",
    )

    args = parser.parse_args()

    prompt_path = args.prompt.expanduser().resolve()

    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    output_path = prompt_path.parent / "codex_repair.cpp"
    codex_log_path = prompt_path.parent / "codex_exec.log"

    command = [
        "codex",
        "exec",
        "-m",
        args.model,
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(output_path),
        "-",
    ]

    prompt_text = prompt_path.read_text(encoding="utf-8")

    print(f"Prompt: {prompt_path}")
    print(f"Model: {args.model}")
    print("Requesting repair from Codex...")

    try:
        process = subprocess.run(
            command,
            input=prompt_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        raise SystemExit("Codex CLI was not found in PATH.")

    codex_log_path.write_text(
        process.stdout or "",
        encoding="utf-8",
    )

    if process.returncode != 0:
        raise SystemExit(
            f"Codex failed with return code {process.returncode}.\n"
            f"See: {codex_log_path}"
        )

    if not output_path.is_file():
        raise SystemExit(
            f"Codex did not create the expected output: {output_path}"
        )

    print(f"Proposed repair: {output_path}")
    print(f"Codex log: {codex_log_path}")


if __name__ == "__main__":
    main()
