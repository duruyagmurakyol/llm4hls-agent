#!/usr/bin/env python3

"""Run one token-efficient structured HLS repair through a direct model API."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.siliconflow import complete, extract_source_response  # noqa: E402


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root: Path, files: list[str]) -> dict[str, str]:
    return {name: sha256(root / name) for name in files}


def validate(config: dict[str, Any], workspace: Path) -> tuple[int, str]:
    host = config["host_validation"]
    compiled = run([str(x) for x in host["command"]], workspace)
    output = compiled.stdout or ""
    if compiled.returncode != 0:
        return compiled.returncode, output
    executed = run([str(x) for x in host["run_command"]], workspace)
    return executed.returncode, output + (executed.stdout or "")


def failure_class(output: str) -> str:
    lower = output.lower()
    if "undefined reference" in lower or "linker" in lower:
        return "interface_or_link"
    if "error:" in lower or "expected" in lower and "before" in lower:
        return "compile"
    if "fail index=" in lower or "expected=" in lower and "actual=" in lower:
        return "functional"
    return "unknown"


def concise_evidence(output: str, limit: int = 1200) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    selected = [
        line
        for line in lines
        if any(
            token in line.lower()
            for token in ("error", "undefined", "fail", "expected", "actual")
        )
    ]
    text = "\n".join(selected[-12:] or lines[-12:])
    return text[-limit:]


def prompt(
    config: dict[str, Any], workspace: Path, evidence: str, category: str
) -> tuple[str, str]:
    editable = str(config["editable_files"][0])
    source = (workspace / editable).read_text(encoding="utf-8")
    contexts: list[str] = []
    for name in config.get("context_files", config["protected_files"]):
        contexts.append(
            f"FILE: {name}\n```\n{(workspace / name).read_text(encoding='utf-8')}\n```"
        )

    system = (
        "You repair AMD/Xilinx HLS C++ code. Return only the complete repaired contents "
        "of the editable source file. Do not use Markdown fences, explanations, JSON, or patches. "
        "Preserve the declared top-function interface and make the smallest necessary repair."
    )
    user = (
        f"Failure class: {category}\n"
        f"Failure evidence:\n{evidence}\n\n"
        f"EDITABLE FILE: {editable}\n```\n{source}\n```\n\n"
        + "\n\n".join(contexts)
        + "\n\nReturn only the full repaired editable file."
    )
    return system, user


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("repair_mode") != "direct_api":
        raise SystemExit("Config must use repair_mode=direct_api")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = str(config["experiment_id"])
    run_dir = root / "runs" / "experiments" / experiment_id / timestamp
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True)
    benchmark_source = root / config["benchmark_source"]
    if not benchmark_source.is_dir():
        raise SystemExit(f"Benchmark source not found: {benchmark_source}")
    shutil.copytree(benchmark_source, workspace)
    metadata = workspace / "fault.txt"
    if metadata.exists():
        metadata.unlink()

    editable = [str(x) for x in config["editable_files"]]
    protected = [str(x) for x in config["protected_files"]]
    tracked = editable + protected
    before_hashes = hashes(workspace, tracked)
    before_source = (workspace / editable[0]).read_text(encoding="utf-8")
    (run_dir / "before.cpp").write_text(before_source, encoding="utf-8")

    pre_code, pre_output = validate(config, workspace)
    (run_dir / "host_validation_before.log").write_text(pre_output, encoding="utf-8")
    category = failure_class(pre_output)
    evidence = concise_evidence(pre_output)
    system_prompt, user_prompt = prompt(config, workspace, evidence, category)
    (run_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (run_dir / "prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")

    thinking_budget_value = config.get("thinking_budget")
    response = complete(
        model=str(config["model"]),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_output_tokens", 2048)),
        timeout_seconds=int(config.get("api_timeout_seconds", 120)),
        thinking_budget=(
            int(thinking_budget_value) if thinking_budget_value is not None else None
        ),
    )
    repaired = extract_source_response(response.content)
    (run_dir / "raw_response.txt").write_text(response.content, encoding="utf-8")
    (run_dir / "api_response.json").write_text(
        json.dumps(response.raw_response, indent=2) + "\n", encoding="utf-8"
    )
    (workspace / editable[0]).write_text(repaired, encoding="utf-8")

    after_hashes = hashes(workspace, tracked)
    modified = [name for name in tracked if before_hashes[name] != after_hashes[name]]
    protected_unchanged = all(
        before_hashes[name] == after_hashes[name] for name in protected
    )
    scope_ok = set(modified).issubset(set(editable))
    diff = "".join(
        difflib.unified_diff(
            before_source.splitlines(keepends=True),
            repaired.splitlines(keepends=True),
            fromfile=f"before/{editable[0]}",
            tofile=f"after/{editable[0]}",
        )
    )
    (run_dir / "repair.diff").write_text(diff, encoding="utf-8")
    changed_lines = sum(
        1
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )

    post_code, post_output = validate(config, workspace)
    (run_dir / "host_validation_after.log").write_text(post_output, encoding="utf-8")

    independent = config["independent_validation"]
    independent_code: int | None = None
    independent_output = ""
    if independent.get("enabled", False):
        command = [
            str(x).replace("{workspace}", str(workspace))
            for x in independent["command"]
        ]
        process = run(command, root)
        independent_code = process.returncode
        independent_output = process.stdout or ""
        (run_dir / "independent_validation.log").write_text(
            independent_output, encoding="utf-8"
        )

    result = {
        "schema_version": 3,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "repair_mode": "direct_api",
        "provider": "siliconflow",
        "model": config["model"],
        "thinking_budget": thinking_budget_value,
        "failure_class": category,
        "pre_host_validation_passed": pre_code == 0,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "tokens_used": response.total_tokens,
        "latency_seconds": response.latency_seconds,
        "modified_files": modified,
        "protected_files_unchanged": protected_unchanged,
        "editable_scope_respected": scope_ok,
        "changed_line_count": changed_lines,
        "tokens_per_changed_line": (
            response.total_tokens / changed_lines
            if response.total_tokens is not None and changed_lines
            else None
        ),
        "post_host_validation_passed": post_code == 0,
        "independent_validation_passed": (
            independent_code == 0 if independent_code is not None else None
        ),
        "repair_diff_present": bool(diff.strip()),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    if not args.keep_workspace:
        shutil.rmtree(workspace)

    print(f"Experiment: {experiment_id}")
    print(f"Model: {config['model']}")
    print(f"Failure class: {category}")
    print(
        f"Tokens: {response.total_tokens} "
        f"(input={response.input_tokens}, output={response.output_tokens})"
    )
    print(f"Latency: {response.latency_seconds:.2f}s")
    print(f"Modified files: {', '.join(modified) if modified else 'none'}")
    print(f"Post-repair host test passed: {post_code == 0}")
    print(f"Independent validation passed: {result['independent_validation_passed']}")
    print(f"Results: {run_dir.relative_to(root)}")

    passed = (
        pre_code != 0
        and scope_ok
        and protected_unchanged
        and post_code == 0
        and result["independent_validation_passed"] is True
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
