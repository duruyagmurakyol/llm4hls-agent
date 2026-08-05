"""Run external tools consistently and capture their output."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    return_code: int | None
    output: str
    cwd: str = ""
    environment: dict[str, str] | None = None
    timed_out: bool = False
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    elapsed_seconds: float = 0.0
    exception: str | None = None

    @property
    def passed(self) -> bool:
        return self.return_code == 0 and not self.timed_out and self.exception is None


def _terminate_process_group(
    process: subprocess.Popen[str],
    grace_seconds: float = 5.0,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _output_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    echo: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CommandResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    rendered = tuple(str(item) for item in command)
    working_directory = str(cwd.resolve())
    environment = dict(env) if env is not None else None
    if echo:
        print("Command:", " ".join(rendered), flush=True)

    started = time.monotonic()
    output = ""
    return_code: int | None = None
    timed_out = False
    exception: str | None = None

    try:
        process = subprocess.Popen(
            rendered,
            cwd=working_directory,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            exception = f"TimeoutExpired: command exceeded {timeout_seconds} seconds"
            partial = _output_text(error.output)
            _terminate_process_group(process)
            tail, _ = process.communicate()
            tail = _output_text(tail)
            output = tail if tail.startswith(partial) else partial + tail
        return_code = process.returncode
    except OSError as error:
        exception = f"{type(error).__name__}: {error}"

    elapsed_seconds = round(time.monotonic() - started, 3)
    output = output or ""
    if timed_out:
        output += f"\nTIMEOUT: command exceeded {timeout_seconds} seconds.\n"
    elif exception:
        output += f"\nERROR: {exception}\n"

    if echo and output:
        print(output, end="", flush=True)

    return CommandResult(
        command=rendered,
        return_code=return_code,
        output=output,
        cwd=working_directory,
        environment=environment,
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=elapsed_seconds,
        exception=exception,
    )
