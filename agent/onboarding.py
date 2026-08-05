"""Discover an HLS benchmark and generate unified optimisation configuration."""

from __future__ import annotations

import configparser
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


def _add_file_references(path: Path, text: str) -> tuple[list[Path], list[Path]]:
    """Return design and testbench files declared by one candidate TCL."""

    design_files: list[Path] = []
    testbenches: list[Path] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = _tokens(line)
        if not tokens or tokens[0] != "add_files":
            continue

        is_tb = "-tb" in tokens
        positional: list[str] = []
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
            resolved = _resolve_tcl_path(token, path)
            if resolved.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            (testbenches if is_tb else design_files).append(resolved)

    return design_files, testbenches


def _tcl_score(path: Path, text: str) -> tuple[int, list[str]]:
    """Score likely baseline synthesis scripts and explain the decision."""

    reasons: list[str] = []
    score = 0
    required = ("set_top", "set_part", "create_clock", "csynth_design")
    for marker in required:
        if marker in text:
            score += 10
            reasons.append(marker)

    lower_name = path.name.lower()
    lower_parts = {part.lower() for part in path.parts}
    if "add_files -tb" in text or re.search(r"add_files\s+.*-tb", text):
        score += 8
        reasons.append("testbench")
    if "add_files" in text:
        score += 3
        reasons.append("design_files")
    if "baseline" in lower_name:
        score += 6
        reasons.append("baseline_name")
    if lower_name.startswith("run_candidate_"):
        score += 4
        reasons.append("candidate_run")
    if "scripts" in lower_parts:
        score += 1
        reasons.append("scripts_dir")

    generated_markers = ("diagnosis", "diagnostic", "hierarchy", "analysis", "report")
    for marker in generated_markers:
        if marker in lower_name:
            score -= 20
            reasons.append(f"penalty:{marker}")
    if "experiments" in lower_parts or "tmp" in lower_parts:
        score -= 5
        reasons.append("penalty:generated_dir")

    # A tracked TCL may point at a locally generated candidate that is absent in
    # a clean checkout. Such a script must never outrank a reproducible baseline.
    design_files, testbenches = _add_file_references(path, text)
    if not design_files:
        score -= 100
        reasons.append("penalty:no_design_source")
    if not testbenches:
        score -= 100
        reasons.append("penalty:no_testbench_source")
    for referenced in [*design_files, *testbenches]:
        if not referenced.is_file():
            score -= 100
            reasons.append(f"penalty:missing_file:{referenced}")

    return score, reasons


def _choose_tcl(root: Path) -> Path:
    files = sorted(root.rglob("*.tcl"))
    if not files:
        raise ValueError(f"No HLS TCL file found under {root}")

    scored: list[tuple[int, Path, list[str]]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        score, reasons = _tcl_score(path, text)
        scored.append((score, path, reasons))

    best_score = max(score for score, _, _ in scored)
    best = [(path, reasons) for score, path, reasons in scored if score == best_score]
    if best_score < 30 or len(best) != 1:
        detail = "; ".join(
            f"{path} score={score} ({', '.join(reasons)})"
            for score, path, reasons in sorted(
                scored,
                key=lambda item: (-item[0], str(item[1])),
            )
        )
        raise ValueError(f"Could not uniquely choose an HLS synthesis TCL file: {detail}")
    return best[0][0]


def _discover_from_task_cfg(root: Path, cfg_path: Path) -> DiscoveredBenchmark:
    parser = configparser.ConfigParser()
    parser.read(cfg_path, encoding="utf-8")

    if "hls" not in parser:
        raise ValueError(f"Missing [hls] section in {cfg_path}")

    hls = parser["hls"]
    required = ("syn.file", "syn.top", "tb.file", "part", "clock")
    missing = [key for key in required if not hls.get(key, "").strip()]
    if missing:
        raise ValueError(
            f"Missing required task.cfg fields: {', '.join(missing)}"
        )

    source = (root / hls["syn.file"].strip()).resolve()
    testbench = (root / hls["tb.file"].strip()).resolve()

    if not source.is_file():
        raise ValueError(f"Configured design source does not exist: {source}")
    if not testbench.is_file():
        raise ValueError(f"Configured testbench does not exist: {testbench}")

    clock_match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*ns\s*",
        hls["clock"],
        re.IGNORECASE,
    )
    if not clock_match:
        raise ValueError(
            f"Unsupported task.cfg clock value: {hls['clock']}"
        )

    headers = tuple(
        sorted(
            path
            for path in source.parent.iterdir()
            if path.is_file() and path.suffix.lower() in HEADER_SUFFIXES
        )
    )

    benchmark_name = root.parent.name if root.name == "golden" else root.name

    return DiscoveredBenchmark(
        name=benchmark_name,
        root=root,
        tcl=cfg_path,
        source=source,
        testbenches=(testbench,),
        headers=headers,
        top_function=hls["syn.top"].strip(),
        part=hls["part"].strip(),
        clock_period_ns=float(clock_match.group(1)),
        project_dir=(
            REPO_ROOT
            / "experiments"
            / "onboarding"
            / benchmark_name
            / "baseline_project"
        ),
    )


def discover_benchmark(root: Path) -> DiscoveredBenchmark:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Benchmark directory not found: {root}")
    tcl_files = sorted(root.rglob("*.tcl"))
    task_cfg = root / "task.cfg"

    if not tcl_files:
        if task_cfg.is_file():
            return _discover_from_task_cfg(root, task_cfg)
        raise ValueError(
            f"No HLS TCL or task.cfg build configuration found under {root}"
        )

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
            positional = [
                _clean(token)
                for token in tokens[1:]
                if not token.startswith("-")
            ]
            if positional:
                project = Path(positional[-1])
                projects.append(
                    project
                    if project.is_absolute()
                    else (tcl.parent / project).resolve()
                )
        elif command == "add_files":
            is_tb = "-tb" in tokens
            positional: list[str] = []
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

    source = _single(
        [
            path
            for path in design_files
            if path.suffix.lower() in SOURCE_SUFFIXES
        ],
        "design source",
    )
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
        headers=tuple(
            dict.fromkeys(path for path in headers if path.is_file())
        ),
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
    optimisation_path.write_text(
        json.dumps(optimisation, indent=2) + "\n",
        encoding="utf-8",
    )

    task = {
        "task_id": f"auto_{benchmark.name}_001",
        "task_kind": "correct_unoptimised",
        "artifacts": {
            "source": _repo_relative(benchmark.source),
            "testbench": [
                _repo_relative(path) for path in benchmark.testbenches
            ],
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
    task_path.write_text(
        json.dumps(task, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Automatic benchmark onboarding")
    print(f"Benchmark: {benchmark.name}")
    print(f"Build configuration: {_repo_relative(benchmark.tcl)}")
    print(f"Top function: {benchmark.top_function}")
    print(f"Source: {_repo_relative(benchmark.source)}")
    print(f"Testbench files: {len(benchmark.testbenches)}")
    print(f"Part: {benchmark.part}")
    print(f"Clock: {benchmark.clock_period_ns:g} ns")
    print(f"Generated task: {_repo_relative(task_path)}")
    print(f"Generated optimisation config: {_repo_relative(optimisation_path)}")
    return task_path
