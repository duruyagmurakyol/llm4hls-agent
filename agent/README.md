# `agent/` package

This directory contains the reusable implementation of the unified LLM4HLS agent. Code here must remain independent of individual benchmark names and directory layouts.

## Package responsibilities

```text
agent/
├── controller.py        top-level task dispatch and unified result writing
├── config.py            task-manifest loading and validation
├── onboarding.py        benchmark discovery and generated configuration
├── onboarding_safe.py   provenance-aware onboarding wrapper
├── state.py             shared metrics, events and result records
├── workspace.py         isolated workspace management
├── analysis/            Vitis hierarchy and source-cause analysis
├── optimise/            PPA optimisation state machine
├── providers/           model API adapters
├── repair/              correctness-repair workflow
└── tools/               shared validation, command and synthesis utilities
```

## Execution path

`run_agent.py` calls `agent.controller.run_agent()`. The controller loads a unified task manifest and dispatches according to `adapter.kind`:

- `autonomous_ppa`: run the budgeted optimisation state machine.
- `legacy_ppa`: accepted as a compatibility alias for the optimisation path.
- `direct_api_repair`: run correctness repair and synthesise the successfully repaired candidate.

Every path returns an `AgentResult`, which is serialised as `unified_agent_result.json`.

## Onboarding

Onboarding converts a benchmark directory into the two configurations required by the controller:

```text
benchmark directory
      |
      +--> task.json
      +--> optimisation.json
      +--> onboarding_report.json
```

Supported build descriptions include:

- HLS TCL scripts containing the top, files, part and clock;
- Vitis `task.cfg` files with an `[hls]` section.

The provenance-aware onboarding layer should prefer a source located inside the requested benchmark and record why that source was selected. Generated paths must remain inside `experiments/onboarding/`.

## `optimise/`

The optimisation package owns the complete PPA candidate lifecycle.

### `runner.py`

Coordinates the resumable workflow and budgets. It decides which stage is next based on existing artefacts rather than assuming a fresh run.

Typical stages are:

1. ensure baseline synthesis;
2. diagnose the baseline hierarchy;
3. map the selected report target to source;
4. prepare a constrained prompt;
5. generate a candidate;
6. run static validation;
7. reject duplicates;
8. run CSim;
9. run synthesis;
10. evaluate and record the verdict.

### `diagnose.py`

Turns report-level evidence into a source-aware prompt. It should preserve already-good regions and focus changes on diagnosed bottlenecks.

### `generate.py`

Calls the configured model provider once per permitted generation attempt, extracts compilable C/C++, and records token usage and latency.

### `evaluate.py`

Extracts candidate outcomes and compares them with baseline metrics. A candidate must remain correct and improve the configured objective; a local loop improvement does not automatically imply acceptance.

### `duplicate.py`

Normalises source text and prevents spending CSim or synthesis calls on equivalent candidates.

### `strategies.json`

Contains benchmark-independent optimisation guidance. Do not add benchmark names, fixed dimensions, or source-specific patches here.

## `repair/`

The repair workflow handles designs that are not yet correct. It classifies observed failures, produces a constrained repair prompt, generates a replacement source, and validates it against the supplied build and testbench.

After host and independent validation pass, the controller synthesises the exact repaired source retained in the run workspace. The repair task is successful only when that synthesis also completes and produces a valid top-level synthesis report. The synthesis call is charged before Vitis is invoked, including failed or timed-out attempts.

The repair path must not silently weaken the testbench or change the public top-level contract.

## `analysis/`

This package interprets Vitis reports and source structure. Its outputs provide evidence such as:

- module and loop hierarchy;
- achieved initiation interval;
- latency and trip count;
- memory-port limits;
- recurrence or timing constraints;
- protected regions that are already near their lower bound.

Analysis should report insufficient evidence explicitly rather than inventing a cause.

## `tools/`

### `synthesis.py`

Owns portable Vitis execution. It:

- reads supported TCL or `task.cfg` build descriptions;
- creates isolated internal TCL scripts;
- replaces only the design source with the current candidate;
- preserves headers, include flags and testbench files;
- runs CSim and synthesis through the shared command runner;
- extracts XML synthesis metrics;
- writes machine-readable reports.

`run_csim(task, candidate)` is the manifest-based CSim boundary. It returns pass status, timeout and return-code information, failure classification and evidence, command, duration, log path and the candidate SHA-256. The controller consumes this structured result rather than parsing terminal output.

`run_synthesis(task, candidate)` is the equivalent synthesis boundary. It requires a successful Vitis process and the expected top-level `*_csynth.xml`, then returns the same process and provenance metadata together with top-level and hierarchical synthesis metrics. Results are stored under `output_dir/synthesis/<candidate-hash>/` and the temporary project is isolated from the benchmark.

The older `run_candidate_csim()` and `run_candidate_synthesis()` entry points remain for the current optimisation workflow, but they use the same shared subprocess and report parsing boundaries.

### `cosim.py`

`run_cosim(task, candidate)` runs `csynth_design` followed by `cosim_design` in a fresh temporary project while reusing the same task parsing and shared process runner as CSim and synthesis.

It returns structured pass, timeout, return-code, duration, command and candidate-provenance fields. Failures distinguish RTL/testbench compilation errors, simulation mismatches, deadlocks, timeouts, missing reports and other co-simulation failures. Only the generated Vitis report files are copied into `output_dir/cosim/<candidate-hash>/reports/`.

Baseline source files and testbenches must never be modified.

### `validation.py`

Runs cheap checks before Vitis. Current checks include unsafe or contradictory HLS pragma structures, interface-array partition risks, required function preservation and failure classification.

A validation rule should be generic and accompanied by a regression test.

### `command_runner.py`

Provides the shared subprocess boundary for external tools. Each result records:

- the rendered command;
- resolved working directory;
- the explicit environment supplied by the caller, or inherited-environment status;
- combined standard output and standard error;
- timeout and elapsed duration;
- timeout status and return code;
- process-start or timeout exception information.

Commands use a 300-second default timeout. A timeout terminates the complete process group with `SIGTERM`, followed by `SIGKILL` if the grace period expires. Existing callers can continue using `command`, `return_code`, `output` and `passed` while structured adapters consume the additional metadata.

Vitis CSim, synthesis and co-simulation reuse this runner; there is no separate low-level Vitis process implementation.

### `reports.py`

Centralises JSON loading and writing so generated evidence remains consistent.

## Provider boundary

`providers/` isolates model-specific HTTP details from the rest of the agent. Core code should depend on a small provider interface rather than SiliconFlow response details.

Provider implementations are responsible for:

- authentication;
- request construction;
- timeout and API error handling;
- response text extraction;
- token and latency metadata.

## Rules for adding code

1. Do not branch on benchmark names in core logic.
2. Do not hard-code array sizes or loop labels from a single benchmark.
3. Do not modify baseline sources or testbenches.
4. Write outputs under the task's configured output directory.
5. Record every important decision as JSON evidence.
6. Put expensive work behind static and functional checks.
7. Add or update tests for new generic behaviour.
8. Keep `scripts/run_agent.py` as the only public orchestration command.

## Validation

From the repository root:

```bash
python -m pytest
python -m compileall -q agent scripts tests
```
