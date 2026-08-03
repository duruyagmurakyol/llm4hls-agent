# `agent/` package

This directory contains the reusable implementation of the unified LLM4HLS agent. Code here must remain independent of individual benchmark names and directory layouts.

## Package responsibilities

```text
agent/
├── baseline.py          verified source and synthesis-report promotion
├── controller.py        top-level task dispatch and unified result writing
├── config.py            task-manifest loading and validation
├── onboarding.py        benchmark discovery and generated configuration
├── onboarding_safe.py   provenance-aware onboarding wrapper
├── state.py             shared phases, metrics, events and result records
├── workspace.py         isolated workspace management
├── analysis/            Vitis hierarchy and source-cause analysis
├── optimise/            PPA optimisation state machine
├── providers/           model API adapters
├── repair/              correctness repair and budget-bounded retries
└── tools/               shared validation, command and synthesis utilities
```

## Execution path

`run_agent.py` calls `agent.controller.run_agent()`. Public competition tasks use `adapter.kind: auto`; the user does not select repair or optimisation.

The controller validates the submitted source in this order:

```text
CSim
→ synthesis
→ C/RTL co-simulation
```

It stops at the first failure:

- CSim failure → repair;
- synthesis failure → repair;
- co-simulation failure → repair;
- all three pass → record a verified initial baseline and enter PPA optimisation.

Each initial tool call is charged before Vitis is invoked. The deciding trajectory event records `route` and `decision_reason`, and the source `task_kind` is metadata rather than routing input.

The route-neutral examples are:

```text
configs/tasks/vector_add_auto_broken.json
configs/tasks/vector_add_auto_correct.json
```

`direct_api_repair`, `autonomous_ppa` and `legacy_ppa` remain accepted for compatibility with existing experiments, but they are not the public automatic interface.

Every path returns an `AgentResult`, which is serialised as `unified_agent_result.json`.

## Verified baseline handoff

After an initially correct source or a repaired source passes CSim, synthesis and C/RTL co-simulation, `baseline.py` promotes the exact verified bytes into:

```text
output_dir/active_baseline.cpp
output_dir/verified_baseline.json
output_dir/verified_baseline_project/solution1/syn/report/*_csynth.xml
```

The metadata record contains the source origin, SHA-256, copied report paths, top-level synthesis metrics and the three validation outcomes. Promotion refuses mismatched source, synthesis and co-simulation hashes.

The PPA runner receives this promoted source, metrics and report tree as its baseline. Its existing baseline initialiser therefore sees cached synthesis reports and does not repeat baseline CSim or synthesis. Cheap diagnosis and prompt artefacts are invalidated when the promoted baseline identity changes, but candidate history is not deleted.

For `adapter.kind: auto`, a successful repair and PPA optimisation now happen in the same invocation when model and validation budget remains. If no candidate budget remains, the run terminates successfully with `status: verified_baseline` and `termination_reason: verified_baseline_no_ppa_budget`.

Explicit `direct_api_repair` tasks retain their repair-only behaviour for compatibility.

## Agent phases

`state.py` defines the explicit `AgentPhase` values used by the unified controller:

```text
discover
validate_initial
diagnose
repair
establish_baseline
diagnose_ppa
generate_optimisation
validate_candidate
select_best
terminate
```

The workflow trajectory remains the detailed record of tool and model stages. Immediately before writing the unified result, the controller derives a separate ordered `phase_transitions` list from that trajectory. Each transition records its previous phase, next phase, reason and relevant evidence such as candidate hash, failure class, return code, timeout status or selected route.

`current_phase` records the final controller phase. Completed and failed runs both finish in `terminate`; the result status and termination reason explain the outcome.

## Onboarding

Onboarding converts a benchmark directory into the configurations required by the controller:

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

For automatic tasks, step 1 normally resolves from the promoted cached reports rather than invoking Vitis again.

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

`runner.py` performs one isolated repair attempt. `retry.py` repeats that attempt while iteration, model-call and independent-CSim budget remain:

```text
diagnose
→ generate repair
→ host validation
→ independent Vitis CSim
→ feed failure into the next prompt
→ retry
```

A retry starts from the previous candidate rather than the original faulty source. Each attempt keeps its own prompt, response, diff, logs, workspace and result. The run root writes `repair_attempts.json` with the ordered attempts, candidate hashes, failed stages, evidence and aggregate token usage. The final successful attempt directory is returned to the controller, so synthesis and co-simulation continue to use the exact validated source.

Repair attempts stop immediately after host and independent CSim validation pass. If they do not pass, attempts continue until the configured `max_attempts` or the shared iteration, model-call, CSim or token budget is exhausted. Failed and timed-out calls remain charged.

After host and independent validation pass, the controller synthesises the exact repaired source retained in the run workspace. Co-simulation runs only when synthesis succeeds and uses that same repaired source. The repair task is successful only when repair validation, synthesis and C/RTL co-simulation all pass.

For an automatic task, the successful repaired source is then copied into the active baseline and passed directly into PPA optimisation. The original benchmark source and testbench remain unchanged.

Synthesis and co-simulation calls are charged before Vitis is invoked, including failed, timed-out or exceptional attempts. Each stage remains in the unified trajectory so a later failure does not discard earlier successful evidence.

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

The repair controller calls this boundary only after synthesis passes, records a `post_repair_cosim` trajectory event, and requires it to pass before returning `fully_verified`.

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
