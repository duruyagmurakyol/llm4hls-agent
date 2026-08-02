#!/usr/bin/env python3

"""Run one structured-feedback HLS repair experiment."""

from __future__ import annotations

import argparse
import difflib
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_experiment import (
    parse_tokens,
    relative_hashes,
    run_command,
    run_streaming_command,
    write_json,
)


def classify_failure(compile_return_code: int, output: str) -> str:
    """Classify the first host-validation failure into a compact category."""

    lowered = output.lower()
    if compile_return_code != 0:
        if "undefined reference" in lowered:
            return "interface_or_link"
        return "compile"
    return "functional"


def compact_evidence(output: str, max_lines: int = 12) -> str:
    """Keep only the first useful non-empty failure lines."""

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines]) or "Host validation failed without textual output."


def fenced_file(workspace: Path, relative: str) -> str:
    path = workspace / relative
    return f"### {relative}\n```cpp\n{path.read_text(encoding='utf-8')}\n```"


def build_structured_prompt(
    config: dict[str, Any],
    workspace: Path,
    failure_class: str,
    evidence: str,
) -> str:
    """Build a compact prompt containing evidence and the complete allowed context."""

    context_files = [
        *[str(path) for path in config["editable_files"]],
        *[str(path) for path in config["protected_files"]],
    ]
    context = "\n\n".join(fenced_file(workspace, path) for path in context_files)
    editable = ", ".join(str(path) for path in config["editable_files"])
    validation = " ".join(
        shlex.quote(str(part)) for part in config["host_validation"]["command"]
    )
    run_validation = " ".join(
        shlex.quote(str(part)) for part in config["host_validation"]["run_command"]
    )

    return f"""You are repairing one HLS C++ design using structured tool feedback.

Workspace: {workspace}
Failure class: {failure_class}
Editable file(s): {editable}

Failure evidence:
```text
{evidence}
```

Allowed context is supplied below. Do not inspect, search, list, or read any files yourself.
Do not run shell commands, g++, Vitis, git, find, rg, or ls.
Modify only the editable file, make the smallest correct repair, and stop.
The runner will validate independently with:
{validation} && {run_validation}

Return no analysis. After editing, reply with one sentence describing the edit.

{context}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a structured-feedback repair experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("repair_mode") != "structured_feedback":
        raise SystemExit("repair_mode must be structured_feedback")

    source_dir = repository_root / config["benchmark_source"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = str(config["experiment_id"])
    run_dir = repository_root / "runs" / "experiments" / experiment_id / timestamp
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_dir, workspace)
    fault_metadata = workspace / "fault.txt"
    if fault_metadata.exists():
        fault_metadata.unlink()

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    editable_files = [str(path) for path in config["editable_files"]]
    protected_files = [str(path) for path in config["protected_files"]]
    tracked_files = editable_files + protected_files
    before_hashes = relative_hashes(workspace, tracked_files)

    before_dir = run_dir / "before"
    before_dir.mkdir()
    for relative in editable_files:
        destination = before_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / relative, destination)

    host = config["host_validation"]
    pre_compile = run_command(list(host["command"]), workspace)
    pre_output = pre_compile.stdout or ""
    pre_return_code = pre_compile.returncode
    compile_return_code = pre_compile.returncode
    if pre_compile.returncode == 0:
        pre_run = run_command(list(host["run_command"]), workspace)
        pre_output += pre_run.stdout or ""
        pre_return_code = pre_run.returncode
    (run_dir / "host_validation_before.log").write_text(pre_output, encoding="utf-8")

    failure_class = classify_failure(compile_return_code, pre_output)
    evidence = compact_evidence(pre_output)
    prompt = build_structured_prompt(config, workspace, failure_class, evidence)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    timeout_seconds = int(config.get("agent_timeout_seconds", 180))
    command = [
        "codex", "exec", "-m", str(config["model"]),
        "--sandbox", "workspace-write", prompt,
    ]
    print(f"Starting structured agent (timeout: {timeout_seconds}s)...", flush=True)
    agent_return_code, agent_output, agent_timed_out = run_streaming_command(
        command, repository_root, run_dir / "agent.log", timeout_seconds
    )

    after_hashes = relative_hashes(workspace, tracked_files)
    modified_files = [
        path for path in tracked_files if before_hashes[path] != after_hashes[path]
    ]
    protected_unchanged = all(
        before_hashes[path] == after_hashes[path] for path in protected_files
    )
    scope_respected = set(modified_files).issubset(set(editable_files))

    after_dir = run_dir / "after"
    after_dir.mkdir()
    diff_parts: list[str] = []
    changed_lines = 0
    for relative in editable_files:
        before_path = before_dir / relative
        after_path = workspace / relative
        destination = after_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(after_path, destination)
        before_lines = before_path.read_text(encoding="utf-8").splitlines(keepends=True)
        after_lines = after_path.read_text(encoding="utf-8").splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            before_lines, after_lines,
            fromfile=f"before/{relative}", tofile=f"after/{relative}",
        ))
        diff_parts.extend(diff)
        changed_lines += sum(
            1 for line in diff if (line.startswith("+") or line.startswith("-"))
            and not line.startswith("+++") and not line.startswith("---")
        )
    repair_diff = "".join(diff_parts)
    (run_dir / "repair.diff").write_text(repair_diff, encoding="utf-8")

    post_compile = run_command(list(host["command"]), workspace)
    post_output = post_compile.stdout or ""
    post_return_code = post_compile.returncode
    if post_compile.returncode == 0:
        post_run = run_command(list(host["run_command"]), workspace)
        post_output += post_run.stdout or ""
        post_return_code = post_run.returncode
    (run_dir / "host_validation_after.log").write_text(post_output, encoding="utf-8")

    independent = config["independent_validation"]
    independent_return_code: int | None = None
    if independent.get("enabled", False):
        independent_command = [
            str(part).replace("{workspace}", str(workspace))
            for part in independent["command"]
        ]
        process = run_command(independent_command, repository_root)
        independent_return_code = process.returncode
        (run_dir / "independent_validation.log").write_text(
            process.stdout or "", encoding="utf-8"
        )

    tokens = parse_tokens(agent_output)
    result: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "config": str(config_path.relative_to(repository_root)),
        "benchmark_source": str(source_dir.relative_to(repository_root)),
        "repair_mode": "structured_feedback",
        "model": config["model"],
        "failure_class": failure_class,
        "evidence_lines": len(evidence.splitlines()),
        "agent_timeout_seconds": timeout_seconds,
        "agent_timed_out": agent_timed_out,
        "pre_host_validation_passed": pre_return_code == 0,
        "agent_return_code": agent_return_code,
        "tokens_used": tokens,
        "modified_files": modified_files,
        "changed_line_count": changed_lines,
        "tokens_per_changed_line": round(tokens / changed_lines, 2)
        if tokens is not None and changed_lines else None,
        "protected_files_unchanged": protected_unchanged,
        "editable_scope_respected": scope_respected,
        "post_host_validation_passed": post_return_code == 0,
        "independent_validation_enabled": independent.get("enabled", False),
        "independent_validation_return_code": independent_return_code,
        "independent_validation_passed": independent_return_code == 0
        if independent_return_code is not None else None,
        "repair_diff_present": bool(repair_diff.strip()),
    }
    write_json(run_dir / "result.json", result)

    if not args.keep_workspace:
        shutil.rmtree(workspace)

    print(f"Experiment: {experiment_id}")
    print(f"Results: {run_dir.relative_to(repository_root)}")
    print(f"Failure class: {failure_class}")
    print(f"Tokens used: {tokens}")
    print(f"Modified files: {', '.join(modified_files) if modified_files else 'none'}")
    print(f"Post-repair host test passed: {post_return_code == 0}")
    print(f"Independent validation passed: {independent_return_code == 0}")

    successful = (
        pre_return_code != 0
        and not agent_timed_out
        and agent_return_code == 0
        and scope_respected
        and protected_unchanged
        and post_return_code == 0
        and independent_return_code == 0
    )
    raise SystemExit(0 if successful else 1)


if __name__ == "__main__":
    main()
