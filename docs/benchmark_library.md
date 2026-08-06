# Benchmark library and overnight suite

The benchmark library separates **runnable adapted tasks** from **upstream source repositories**. A repository containing useful HLS code is not automatically safe to execute through the agent: it must expose a stable top function, editable source, protected testbench and a Vitis HLS build description that the onboarding layer can resolve.

## Runnable collections

`configs/suites/overnight_full.json` currently discovers two collections:

1. `external/fpt26-harness/tasks/*`: the three supplied Track-A reference packages.
2. `benchmarks/`: every locally adapted benchmark for which automatic onboarding can resolve a `task.cfg` or TCL HLS flow.

The second collection includes the project's adapted HLS-Eval, vector-add, ATAX, BICG and other compatible benchmark directories without requiring each task to be repeated in the suite file.

The suite is deliberately sequential. Parallel Vitis synthesis and co-simulation would make process failures, temporary projects and credit accounting harder to attribute to one task.

## Upstream provenance

The source registry in the suite records repositories and revisions separately from the runnable task list.

- **HLS-Eval** provides 94 LLM-ready HLS designs drawn from benchmark families including PolyBench, MachSuite, CHStone, Rosetta and others. The locally adapted subset is the main external corpus used tonight.
- **MachSuite** is represented through HLS-Eval adaptations. Raw MachSuite directories are not executed directly because their build and validation interfaces differ from the agent contract.
- **Bench4HLS** is recorded as a future import candidate. It is not included until task compatibility and licensing are audited.
- **Track-A reference tasks** come from the supplied archive and remain internal unless redistribution permission is clear.

A source being publicly visible is not enough to make it redistributable. Before creating the competition archive, retain only tasks whose upstream licence and attribution have been recorded.

## List the runnable library

```bash
python3 scripts/run_task_suite.py --list
```

Filter by task ID, path, collection, source or tag:

```bash
python3 scripts/run_task_suite.py --list --only hls-eval
python3 scripts/run_task_suite.py --list --only track_a --skip '*residual*'
```

The list command does not call a model or Vitis.

## Safe test before an overnight run

Run one discovered task with a fresh result directory:

```bash
python3 scripts/run_task_suite.py \
  --fresh \
  --max-tasks 1 \
  --run-id overnight_smoke
```

The runner performs directory onboarding first, then runs the public agent command. It writes:

```text
results/suites/<run-id>/
├── suite_definition.json
├── suite_state.json
├── suite_summary.csv
├── logs/
├── preflight/
└── archived_outputs/       # only with --fresh and existing outputs
```

Both `suite_state.json` and `suite_summary.csv` are replaced atomically after every task.

## Full clean overnight run

Start this only after any current `run_agent.py`, `vitis-run`, `vitis_hls` or `xsim` process has finished. The runner refuses concurrent execution by default.

```bash
python3 scripts/run_task_suite.py \
  --fresh \
  --clear-vitis-cache
```

`--fresh` moves each task's previous output into the suite's `archived_outputs/` directory before executing it. `--clear-vitis-cache` removes `/tmp/llm4hls-agent` once at suite start so the run does not silently reuse an earlier temporary Vitis project.

To keep the machine awake and preserve the terminal session on Linux:

```bash
tmux new -s llm4hls-overnight
systemd-inhibit --what=sleep --why='LLM4HLS overnight suite' \
  python3 scripts/run_task_suite.py --fresh --clear-vitis-cache
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t llm4hls-overnight
```

## Resume an interrupted suite

```bash
python3 scripts/run_task_suite.py \
  --resume-suite results/suites/<run-id>
```

Completed task IDs are skipped. To retry rows whose `success` field is not true:

```bash
python3 scripts/run_task_suite.py \
  --resume-suite results/suites/<run-id> \
  --rerun-failed
```

Task-level `--resume` is separate and is enabled only with `--resume-tasks`. Do not combine `--fresh` and `--resume-tasks` for a controlled experiment.

## Failure policy

The default suite continues after an individual task failure or timeout. Every task has a four-hour process timeout. Use `--stop-on-error` when debugging.

A successful process exit is not treated as sufficient evidence. The CSV also records:

- agent `success`, status and termination reason;
- final-design verification;
- 100 MHz compliance;
- selected candidate and measured hardware metrics;
- token and tool-call consumption;
- reference-harness credits and score estimate;
- exact task log and result paths.

## Adding another source

Do not add a raw GitHub directory directly to the overnight suite. First:

1. record repository, exact commit and licence;
2. place the source under `external/` or an attributed benchmark directory;
3. provide a protected testbench and a Vitis `task.cfg` or supported TCL flow;
4. run `scripts/run_agent.py <directory> --onboard-only`;
5. run a no-model baseline CSim/synthesis check;
6. only then include it under the `benchmarks/` discovery root.

This keeps the library useful for research rather than turning overnight execution into an environment-setup test.
