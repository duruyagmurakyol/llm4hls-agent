"""Discover an HLS benchmark and generate unified optimisation configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp"}


@dataclass(frozen=True)
class DiscoveredBenchmark:
    name: str
    root: Path
    tcl: Path
    source: Path
    testbenches: tuple[Path, ...]
    headers: tuple[Path, ...]
    top_function: str
    part: str
    clock_period_ns: float
    project_dir: Path


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError as error:
        raise ValueError(f"Benchmark file must be inside the repository: {path}") from error


def _tokens(line: str) -> list[str]:
    return re.findall(r'\{[^}]*\}|"[^"]*"|\S+', line)


def _clean(token: str) -> str:
    return token.strip().strip('{}"')


def _resolve_tcl_path(token: str, tcl: Path) -> Path:
    value = _clean(token)
    value = value.replace("$::env(PWD)", str(REPO_ROOT))
    value = value.replace("${::env(PWD)}", str(REPO_ROOT))
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = ((tcl.parent / path).resolve(), (REPO_ROOT / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _single(values: list[Any], description: str) -> Any:
    unique = list(dict.fromkeys(values))
    if len(unique) != 1:
        rendered = ", ".join(str(value) for value in unique) or "none"
        raise ValueError(f"Could not uniquely determine {description}; found: {rendered}")
    return unique[0]


def _choose_tcl(root: Path) -> Path:
    files = sorted(root.rglob("*.tcl"))
    if not files:
        raise ValueError(f"No HLS TCL file found under {root}")
    scored: list[tuple[int, Path]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        score = sum(marker in text for marker in ("set_top", "set_part", "create_clock", "csynth_design"))
        scored.append((score, path))
    best_score = max(score for score, _ in scored)
    best = [path for score, path in scored if score == best_score]
    if best_score < 3 or len(best) != 1:
        raise ValueError(
            "Could not uniquely choose an HLS synthesis TCL file: "
            + ", ".join(str(path) for path in best)
        )
    return best[0]


def discover_benchmark(root: Path) -> DiscoveredBenchmark:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Benchmark directory not found: {root}")
    tcl = _choose_tcl(root)
    lines = tcl.read_text(encoding="utf-8", errors="replace").splitlines()

    tops: list[str] = []
    parts: list[str] = []
    clocks: list[float] = []
    projects: list[Path] = []
    design_files: list[Path] = []
    testbenches: list[Path] = []
    headers: list[Path] = []

    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = _tokens(line)
        command = tokens[0]
        if command == "set_top" and len(tokens) >= 2:
            tops.append(_clean(tokens[1]))
        elif command == "set_part" and len(tokens) >= 2:
            parts.append(_clean(tokens[-1]))
        elif command == "create_clock":
            match = re.search(r"(?:-period\s+)?([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                clocks.append(float(match.group(1)))
        elif command in {"open_project", "open_component"}:
            positional = [_clean(token) for token in tokens[1:] if not token.startswith("-")]
            if positional:
                project = Path(positional[-1])
                projects.append(project if project.is_absolute() else (tcl.parent / project).resolve())
        elif command == "add_files":
            is_tb = "-tb" in tokens
            positional = []
            skip_next = False
            for token in tokens[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if token in {"-cflags", "-csimflags", "-tbflags"}:
                    skip_next = True
                    continue
                if token.startswith("-"):
                    continue
                positional.append(token)
            for token in positional:
                path = _resolve_tcl_path(token, tcl)
                suffix = path.suffix.lower()
                if suffix in HEADER_SUFFIXES:
                    headers.append(path)
                elif suffix in SOURCE_SUFFIXES:
                    (testbenches if is_tb else design_files).append(path)

    source = _single([path for path in design_files if path.suffix.lower() in SOURCE_SUFFIXES], "design source")
    if not source.is_file():
        raise ValueError(f"Discovered design source does not exist: {source}")
    if not testbenches:
        raise ValueError("No testbench was found in the selected TCL via add_files -tb.")
    for path in testbenches:
        if not path.is_file():
            raise ValueError(f"Discovered testbench does not exist: {path}")

    return DiscoveredBenchmark(
        name=root.name,
        root=root,
        tcl=tcl,
        source=source,
        testbenches=tuple(dict.fromkeys(testbenches)),
        headers=tuple(dict.fromkeys(path for path in headers if path.is_file())),
        top_function=_single(tops, "top function"),
        part=_single(parts, "FPGA part"),
        clock_period_ns=_single(clocks, "clock period"),
        project_dir=_single(projects, "baseline project directory"),
    )


def onboard_benchmark(root: Path) -> Path:
    benchmark = discover_benchmark(root)
    generated_dir = REPO_ROOT / "experiments" / "onboarding" / benchmark.name
    generated_dir.mkdir(parents=True, exist_ok=True)
    optimisation_path = generated_dir / "optimisation.json"
    task_path = generated_dir / "task.json"

    output_dir = f"experiments/{benchmark.name}/autonomous_ppa"
    optimisation: dict[str, Any] = {
        "experiment_name": f"{benchmark.name}_ppa",
        "benchmark": benchmark.name,
        "top_function": benchmark.top_function,
        "baseline": {
            "source": _repo_relative(benchmark.source),
            "tcl": _repo_relative(benchmark.tcl),
            "project_dir": _repo_relative(benchmark.project_dir),
        },
        "validation": {
            "constant_loop_tail_bounds": True,
            "preserve_diagnosed_loop_label": True,
        },
        "prompt_constraints": [
            "Preserve the top-level function signature and all testbench-observed semantics.",
            "Do not modify the supplied testbench or baseline source in place.",
        ],
        "output_dir": output_dir,
        "model": {
            "provider": "siliconflow",
            "name": "Qwen/Qwen3.5-122B-A10B",
            "temperature": 0.0,
            "max_tokens": 4096,
            "enable_thinking": False,
        },
        "budget": {"max_candidates": 5, "max_synthesis_calls": 4},
    }
    optimisation_path.write_text(json.dumps(optimisation, indent=2) + "\n", encoding="utf-8")

    task = {
        "task_id": f"auto_{benchmark.name}_001",
        "task_kind": "correct_unoptimised",
        "artifacts": {
            "source": _repo_relative(benchmark.source),
            "testbench": [_repo_relative(path) for path in benchmark.testbenches],
            "headers": [_repo_relative(path) for path in benchmark.headers],
            "build_files": [_repo_relative(benchmark.tcl)],
        },
        "interface": {
            "top_function": benchmark.top_function,
            "language": "cpp",
            "numerical_tolerance": None,
            "protected_contract": [
                "Preserve the top-level function signature.",
                "Preserve output semantics checked by the supplied testbench.",
            ],
        },
        "target": {
            "tool": "AMD Vitis HLS",
            "tool_version": "2025.2",
            "part": benchmark.part,
            "clock_period_ns": benchmark.clock_period_ns,
            "resource_limits": {},
        },
        "budgets": {
            "max_iterations": 5,
            "max_csim_calls": 5,
            "max_cosim_calls": 0,
            "max_synthesis_calls": 4,
            "max_model_calls": 5,
            "max_total_tokens": None,
        },
        "model": optimisation["model"],
        "adapter": {
            "kind": "autonomous_ppa",
            "config": _repo_relative(optimisation_path),
        },
        "output_dir": f"experiments/track_a/auto_{benchmark.name}_001",
    }
    task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")

    print("Automatic benchmark onboarding")
    print(f"Benchmark: {benchmark.name}")
    print(f"Top function: {benchmark.top_function}")
    print(f"Source: {_repo_relative(benchmark.source)}")
    print(f"Testbench files: {len(benchmark.testbenches)}")
    print(f"Part: {benchmark.part}")
    print(f"Clock: {benchmark.clock_period_ns:g} ns")
    print(f"Generated task: {_repo_relative(task_path)}")
    print(f"Generated optimisation config: {_repo_relative(optimisation_path)}")
    return task_path
