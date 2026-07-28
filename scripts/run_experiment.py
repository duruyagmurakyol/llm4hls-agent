#!/usr/bin/env python3

"""Run one reproducible HLS repair experiment from a JSON configuration."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(
    command: list[str],
    cwd: Path,
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture combined stdout/stderr."""

    try:
        return subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Command not found: {command[0]}") from error


def run_streaming_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, str, bool]:
    """Run a command with live output, continuous logging and a hard timeout."""

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            start_new_session=True,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Command not found: {command[0]}") from error

    if process.stdout is None:
        raise SystemExit("Unable to capture agent output")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    chunks: list[bytes] = []
    started = time.monotonic()
    timed_out = False

    with log_path.open("wb") as log_file:
        while True:
            elapsed = time.monotonic() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0 and process.poll() is None:
                timed_out = True
                message = (
                    f"\n[runner] Agent timed out after {timeout_seconds} seconds; "
                    "terminating process group.\n"
                ).encode()
                chunks.append(message)
                log_file.write(message)
                log_file.flush()
                print(message.decode(), end="", flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

            events = selector.select(timeout=max(0.0, min(0.25, remaining)))
            for key, _ in events:
                data = os.read(key.fileobj.fileno(), 4096)
                if data:
                    chunks.append(data)
                    log_file.write(data)
                    log_file.flush()
                    print(data.decode(errors="replace"), end="", flush=True)
                else:
                    selector.unregister(key.fileobj)

            if process.poll() is not None:
                while True:
                    data = os.read(process.stdout.fileno(), 4096)
                    if not data:
                        break
                    chunks.append(data)
                    log_file.write(data)
                    print(data.decode(errors="replace"), end="", flush=True)
                log_file.flush()
                break

    selector.close()
    output = b"".join(chunks).decode(errors="replace")
    return process.returncode if process.returncode is not None else 1, output, timed_out


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_hashes(root: Path, paths: list[str]) -> dict[str, str]:
    """Hash configured files relative to a workspace."""

    hashes: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Configured file does not exist: {path}")
        hashes[relative] = sha256(path)
    return hashes


def parse_tokens(output: str) -> int | None:
    """Extract Codex token usage from its textual output when available."""

    matches = re.findall(r"tokens used\s*\n?\s*([0-9][0-9,]*)", output, re.IGNORECASE)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def build_prompt(config: dict[str, Any], workspace: Path) -> str:
    """Build the autonomous repair prompt from configuration fields."""

    constraints = "\n".join(str(item) for item in config["agent_constraints"])
    return f"Repair {workspace}.\n\n{constraints}\n"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one configuration-driven autonomous HLS repair experiment."
    )
    parser.add_argument("config", type=Path, help="Path to an experiment JSON file")
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the mutable workspace after the run",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parent.parent
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required = {
        "experiment_id",
        "benchmark_source",
        "repair_mode",
        "model",
        "editable_files",
        "protected_files",
        "host_validation",
        "independent_validation",
        "agent_constraints",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise SystemExit(f"Missing configuration keys: {', '.join(missing)}")
    if config["repair_mode"] != "autonomous":
        raise SystemExit("This first runner currently supports repair_mode=autonomous only")

    timeout_seconds = int(config.get("agent_timeout_seconds", 300))
    if timeout_seconds <= 0:
        raise SystemExit("agent_timeout_seconds must be greater than zero")

    source_dir = repository_root / config["benchmark_source"]
    if not source_dir.is_dir():
        raise SystemExit(f"Benchmark source not found: {source_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = str(config["experiment_id"])
    run_dir = repository_root / "results" / "experiments" / experiment_id / timestamp
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_dir, workspace)

    copied_config = run_dir / "config.json"
    copied_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    editable_files = [str(path) for path in config["editable_files"]]
    protected_files = [str(path) for path in config["protected_files"]]
    all_tracked_files = editable_files + protected_files
    before_hashes = relative_hashes(workspace, all_tracked_files)

    before_dir = run_dir / "before"
    before_dir.mkdir()
    for relative in editable_files:
        destination = before_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / relative, destination)

    host_config = config["host_validation"]
    pre_compile = run_command(list(host_config["command"]), workspace)
    pre_output = pre_compile.stdout or ""
    pre_return_code = pre_compile.returncode
    if pre_compile.returncode == 0:
        pre_run = run_command(list(host_config["run_command"]), workspace)
        pre_output += pre_run.stdout or ""
        pre_return_code = pre_run.returncode
    (run_dir / "host_validation_before.log").write_text(pre_output, encoding="utf-8")

    prompt = build_prompt(config, workspace)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    codex_command = [
        "codex",
        "exec",
        "-m",
        str(config["model"]),
        "--sandbox",
        "workspace-write",
        prompt,
    ]
    print(f"Starting agent (timeout: {timeout_seconds}s)...", flush=True)
    agent_return_code, codex_output, agent_timed_out = run_streaming_command(
        codex_command,
        repository_root,
        run_dir / "agent.log",
        timeout_seconds,
    )

    after_hashes = relative_hashes(workspace, all_tracked_files)
    modified_files = [
        relative
        for relative in all_tracked_files
        if before_hashes[relative] != after_hashes[relative]
    ]
    protected_files_unchanged = all(
        before_hashes[relative] == after_hashes[relative]
        for relative in protected_files
    )
    editable_scope_respected = set(modified_files).issubset(set(editable_files))

    after_dir = run_dir / "after"
    after_dir.mkdir()
    diff_parts: list[str] = []
    for relative in editable_files:
        before_path = before_dir / relative
        after_path = workspace / relative
        destination = after_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(after_path, destination)
        before_lines = before_path.read_text(encoding="utf-8").splitlines(keepends=True)
        after_lines = after_path.read_text(encoding="utf-8").splitlines(keepends=True)
        diff_parts.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"before/{relative}",
                tofile=f"after/{relative}",
            )
        )
    repair_diff = "".join(diff_parts)
    (run_dir / "repair.diff").write_text(repair_diff, encoding="utf-8")

    post_compile = run_command(list(host_config["command"]), workspace)
    post_output = post_compile.stdout or ""
    post_return_code = post_compile.returncode
    if post_compile.returncode == 0:
        post_run = run_command(list(host_config["run_command"]), workspace)
        post_output += post_run.stdout or ""
        post_return_code = post_run.returncode
    (run_dir / "host_validation_after.log").write_text(post_output, encoding="utf-8")

    independent_config = config["independent_validation"]
    independent_return_code: int | None = None
    independent_output = ""
    if independent_config.get("enabled", False):
        command = [
            str(part).replace("{workspace}", str(workspace))
            for part in independent_config["command"]
        ]
        independent_process = run_command(command, repository_root)
        independent_return_code = independent_process.returncode
        independent_output = independent_process.stdout or ""
        (run_dir / "independent_validation.log").write_text(
            independent_output,
            encoding="utf-8",
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "config": str(config_path.relative_to(repository_root)),
        "benchmark_source": str(source_dir.relative_to(repository_root)),
        "repair_mode": config["repair_mode"],
        "model": config["model"],
        "agent_timeout_seconds": timeout_seconds,
        "agent_timed_out": agent_timed_out,
        "pre_host_validation_passed": pre_return_code == 0,
        "agent_return_code": agent_return_code,
        "tokens_used": parse_tokens(codex_output),
        "modified_files": modified_files,
        "protected_files_unchanged": protected_files_unchanged,
        "editable_scope_respected": editable_scope_respected,
        "post_host_validation_passed": post_return_code == 0,
        "independent_validation_enabled": independent_config.get("enabled", False),
        "independent_validation_return_code": independent_return_code,
        "independent_validation_passed": independent_return_code == 0
        if independent_return_code is not None
        else None,
        "repair_diff_present": bool(repair_diff.strip()),
        "artifacts": {
            "prompt": "prompt.txt",
            "agent_log": "agent.log",
            "before": "before/",
            "after": "after/",
            "diff": "repair.diff",
            "host_before_log": "host_validation_before.log",
            "host_after_log": "host_validation_after.log",
            "independent_validation_log": "independent_validation.log"
            if independent_config.get("enabled", False)
            else None,
        },
    }
    write_json(run_dir / "result.json", result)

    if not args.keep_workspace:
        shutil.rmtree(workspace)

    print(f"Experiment: {experiment_id}")
    print(f"Results: {run_dir.relative_to(repository_root)}")
    print(f"Agent timed out: {agent_timed_out}")
    print(f"Pre-repair host test passed: {result['pre_host_validation_passed']}")
    print(f"Modified files: {', '.join(modified_files) if modified_files else 'none'}")
    print(f"Protected files unchanged: {protected_files_unchanged}")
    print(f"Post-repair host test passed: {result['post_host_validation_passed']}")
    print(f"Independent validation passed: {result['independent_validation_passed']}")

    successful = (
        not result["pre_host_validation_passed"]
        and not result["agent_timed_out"]
        and result["agent_return_code"] == 0
        and result["editable_scope_respected"]
        and result["protected_files_unchanged"]
        and result["post_host_validation_passed"]
        and result["independent_validation_passed"]
    )
    raise SystemExit(0 if successful else 1)


if __name__ == "__main__":
    main()
