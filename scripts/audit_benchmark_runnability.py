#!/usr/bin/env python3

"""Audit retained LLM4HLS benchmarks and task manifests for runnability.

Static mode is deliberately dependency-light and suitable for CI. It verifies
that every tracked local task manifest loads, every suite-local path exists, and
every benchmark discovered by the canonical ``overnight_full`` registry can be
onboarded by the same discovery code used by the agent.

With ``--vitis`` the audit additionally executes the original source of every
discoverable local benchmark through Vitis CSim. If CSim passes, synthesis is
also executed. A design-level CSim/synthesis failure is *reported* but is not a
runnability failure because many retained repair benchmarks are intentionally
faulty. Exceptions, timeouts, missing paths and malformed configuration are
hard failures.

No model provider or API key is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_task  # noqa: E402
from agent.onboarding_safe import resolve_benchmark  # noqa: E402
from agent.tools.synthesis import find_vitis_run, run_csim, run_synthesis  # noqa: E402
from scripts.run_task_suite import _expand_suite, _load_suite  # noqa: E402


@dataclass
class AuditItem:
    kind: str
    path: str
    status: str
    detail: str = ""
    hard_failure: bool = False


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _external_reference(value: str) -> bool:
    return value.replace("\\", "/").startswith("external/")


def _manifest_references_external(data: dict[str, Any]) -> bool:
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    values: list[str] = []
    for key in ("source", "specification"):
        value = artifacts.get(key)
        if isinstance(value, str):
            values.append(value)
    for key in ("testbench", "headers", "build_files"):
        value = artifacts.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if isinstance(item, str))
    return any(_external_reference(value) for value in values)


def _matches(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    lowered = path.casefold()
    return any(pattern.casefold() in lowered for pattern in patterns)


def _audit_manifests(patterns: list[str]) -> list[AuditItem]:
    items: list[AuditItem] = []
    for path in sorted((REPO_ROOT / "configs" / "tasks").rglob("*.json")):
        relative = _display(path)
        if not _matches(relative, patterns):
            continue
        data = _load_json(path)
        if data is None:
            items.append(
                AuditItem("manifest", relative, "invalid_json", "could not parse JSON", True)
            )
            continue

        # Index/registry JSON files are not task manifests.
        if not {"task_id", "artifacts", "adapter"}.issubset(data):
            items.append(AuditItem("manifest", relative, "registry", "not a task manifest"))
            continue

        if _manifest_references_external(data):
            missing_external = False
            artifacts = data.get("artifacts", {})
            for value in [
                artifacts.get("source"),
                *(artifacts.get("testbench") or []),
                *(artifacts.get("headers") or []),
                *(artifacts.get("build_files") or []),
            ]:
                if isinstance(value, str) and _external_reference(value):
                    resolved = REPO_ROOT / value
                    if not resolved.exists():
                        missing_external = True
            if missing_external:
                items.append(
                    AuditItem(
                        "manifest",
                        relative,
                        "external_dependency",
                        "organiser/evaluator input is not present in this checkout",
                    )
                )
                continue

        try:
            task = load_task(path)
        except Exception as error:  # noqa: BLE001 - audit must report every malformed task
            items.append(AuditItem("manifest", relative, "load_failed", str(error), True))
        else:
            items.append(
                AuditItem(
                    "manifest",
                    relative,
                    "loaded",
                    f"{task.task_id} [{task.adapter_kind}]",
                )
            )
    return items


def _audit_suites(patterns: list[str]) -> list[AuditItem]:
    items: list[AuditItem] = []
    for path in sorted((REPO_ROOT / "configs" / "suites").glob("*.json")):
        relative = _display(path)
        if not _matches(relative, patterns):
            continue
        data = _load_json(path)
        if data is None:
            items.append(AuditItem("suite", relative, "invalid_json", "could not parse JSON", True))
            continue

        schema = data.get("schema_version")
        missing: list[str] = []
        external: list[str] = []
        if schema == 2:
            for task in data.get("tasks", []):
                if not isinstance(task, dict) or not isinstance(task.get("path"), str):
                    continue
                value = str(task["path"])
                resolved = Path(value).expanduser()
                if not resolved.is_absolute():
                    resolved = REPO_ROOT / resolved
                if not resolved.exists():
                    (external if _external_reference(value) else missing).append(value)
        elif schema == 1:
            for collection in data.get("collections", []):
                if not isinstance(collection, dict) or collection.get("enabled", True) is False:
                    continue
                value = collection.get("root")
                if not isinstance(value, str):
                    continue
                resolved = Path(value).expanduser()
                if not resolved.is_absolute():
                    resolved = REPO_ROOT / resolved
                if not resolved.exists():
                    (external if _external_reference(value) else missing).append(value)
        else:
            items.append(
                AuditItem("suite", relative, "unsupported_schema", f"schema_version={schema!r}", True)
            )
            continue

        if missing:
            items.append(
                AuditItem(
                    "suite",
                    relative,
                    "missing_local_paths",
                    ", ".join(sorted(set(missing))),
                    True,
                )
            )
        elif external:
            items.append(
                AuditItem(
                    "suite",
                    relative,
                    "loaded_with_external_dependencies",
                    ", ".join(sorted(set(external))),
                )
            )
        else:
            items.append(AuditItem("suite", relative, "loaded"))
    return items


def _discover_local_benchmarks(patterns: list[str]) -> tuple[list[tuple[str, Path]], list[AuditItem]]:
    suite_path = REPO_ROOT / "configs" / "suites" / "overnight_full.json"
    suite = _load_suite(suite_path)
    discovered = _expand_suite(suite)
    roots: dict[str, Path] = {}
    items: list[AuditItem] = []

    for spec in discovered:
        path = spec.resolved_path
        if not path.exists():
            # Missing organiser packages are intentionally ignored here; suite
            # path accounting is handled separately above.
            continue
        try:
            path.relative_to(REPO_ROOT / "benchmarks")
        except ValueError:
            continue
        relative = _display(path)
        if not _matches(relative, patterns):
            continue
        roots[relative] = path

    for relative, path in sorted(roots.items()):
        try:
            task = resolve_benchmark(path)
        except Exception as error:  # noqa: BLE001
            items.append(AuditItem("benchmark", relative, "onboarding_failed", str(error), True))
        else:
            source = str(task.data["artifacts"]["source"])
            build = str(task.data["artifacts"]["build_files"][0])
            items.append(
                AuditItem(
                    "benchmark",
                    relative,
                    "onboarded",
                    f"top={task.data['interface']['top_function']} source={source} build={build}",
                )
            )
    return sorted(roots.items()), items


def _resolve_artifact(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _audit_vitis(benchmarks: list[tuple[str, Path]]) -> list[AuditItem]:
    items: list[AuditItem] = []
    find_vitis_run()  # fail immediately with the normal actionable error if Vitis is not loaded

    for relative, root in benchmarks:
        try:
            task = resolve_benchmark(root)
            source = _resolve_artifact(str(task.data["artifacts"]["source"]))
            csim = run_csim(task, source)
        except Exception as error:  # noqa: BLE001
            items.append(AuditItem("vitis", relative, "csim_exception", str(error), True))
            continue

        if bool(csim.get("timed_out")):
            items.append(
                AuditItem(
                    "vitis",
                    relative,
                    "csim_timeout",
                    str(csim.get("log_path", "")),
                    True,
                )
            )
            continue

        if csim.get("passed") is not True:
            # This is intentionally not a hard failure: syntax/functional repair
            # benchmarks are expected to fail their submitted source. The
            # important property is that the task reached Vitis and produced a
            # classified result rather than raising a configuration/path error.
            items.append(
                AuditItem(
                    "vitis",
                    relative,
                    "csim_design_failure",
                    f"class={csim.get('failure_class')} log={csim.get('log_path')}",
                )
            )
            continue

        try:
            synthesis = run_synthesis(task, source)
        except Exception as error:  # noqa: BLE001
            items.append(AuditItem("vitis", relative, "synthesis_exception", str(error), True))
            continue

        if bool(synthesis.get("timed_out")):
            items.append(
                AuditItem(
                    "vitis",
                    relative,
                    "synthesis_timeout",
                    str(synthesis.get("log_path", "")),
                    True,
                )
            )
        elif synthesis.get("passed") is True:
            items.append(
                AuditItem(
                    "vitis",
                    relative,
                    "csim_and_synthesis_pass",
                    f"log={synthesis.get('log_path')}",
                )
            )
        else:
            items.append(
                AuditItem(
                    "vitis",
                    relative,
                    "synthesis_design_failure",
                    f"class={synthesis.get('failure_class')} log={synthesis.get('log_path')}",
                )
            )
    return items


def audit_repository(*, vitis: bool = False, patterns: list[str] | None = None) -> dict[str, Any]:
    patterns = list(patterns or [])
    benchmarks, benchmark_items = _discover_local_benchmarks(patterns)
    items = [
        *_audit_manifests(patterns),
        *_audit_suites(patterns),
        *benchmark_items,
    ]
    if vitis:
        items.extend(_audit_vitis(benchmarks))

    hard_failures = [item for item in items if item.hard_failure]
    return {
        "schema_version": 1,
        "mode": "vitis" if vitis else "static",
        "benchmarks_discovered": len(benchmarks),
        "items": [asdict(item) for item in items],
        "hard_failures": [asdict(item) for item in hard_failures],
        "passed": not hard_failures,
    }


def _print_report(report: dict[str, Any]) -> None:
    for item in report["items"]:
        marker = "FAIL" if item["hard_failure"] else "OK"
        detail = f" — {item['detail']}" if item.get("detail") else ""
        print(f"[{marker}] {item['kind']}: {item['path']} :: {item['status']}{detail}")
    print()
    print(f"Discovered local benchmarks: {report['benchmarks_discovered']}")
    print(f"Hard failures: {len(report['hard_failures'])}")
    print(f"Audit result: {'PASS' if report['passed'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit all retained benchmark/manfiest paths and optionally exercise Vitis."
    )
    parser.add_argument(
        "--vitis",
        action="store_true",
        help="also run CSim for every local benchmark and synthesis when CSim passes",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="audit only paths containing this substring; may be repeated",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optionally write the complete audit report as JSON",
    )
    args = parser.parse_args()

    report = audit_repository(vitis=args.vitis, patterns=args.only)
    _print_report(report)

    if args.json_out:
        output = args.json_out.expanduser()
        if not output.is_absolute():
            output = REPO_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {_display(output)}")

    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
