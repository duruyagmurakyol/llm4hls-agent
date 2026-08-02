# Vector-Add Pre-Refactor Baseline

## Purpose

This document records the working behaviour of the vector-add repair and PPA workflows before the Track A architecture is changed.

Generated logs, candidates and Vitis workspaces remain local and are excluded from Git. This document records only the results required for later regression testing.

## Environment

* Date: 3 August 2026
* Branch: `refactor/unified-agent`
* Commit: `<INSERT COMMIT HASH>`
* Vitis HLS: 2025.2
* FPGA part: `xczu3eg-sfvc784-2-e`
* Clock target: 10 ns
* Model: `Qwen/Qwen3.5-122B-A10B`

## Repair Baseline

### Fault

The vector-add implementation incorrectly performed subtraction:

```cpp
c[i] = a[i] - b[i];
```

### Command

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_repair.json
```

### Result

* Failure classification: functional
* Model calls: 1
* Input tokens: 556
* Output tokens: 77
* Total tokens: 633
* Model latency: 2.79 seconds
* Modified file: `src/vector_add.cpp`
* Host validation: PASS
* Vitis CSim: PASS
* Protected files unchanged: PASS

The repaired design was then synthesised manually:

```bash
v++ -c \
  --mode hls \
  --part xczu3eg-sfvc784-2-e \
  --config task.cfg \
  --work_dir vitis_synthesis
```

Synthesis results:

* C synthesis: PASS
* RTL generation: PASS
* Vivado IP packaging: PASS
* Estimated Fmax: 605.69 MHz
* Initiation interval: 1

## PPA Workflow Baseline

### Command

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_track_a.json
```

### Observed behaviour

* The original working design was preserved.
* Candidate 002 passed CSim and synthesis.
* Candidate 002 achieved an estimated Fmax of 136.99 MHz.
* Vitis could not satisfy its requested pipeline because the implementation generated excessive accesses through the same AXI memory port.
* Candidate 002 was rejected because it did not improve latency or initiation interval.
* Candidate 003 used 816 input tokens and 419 output tokens.
* Candidate 003 was rejected by static validation before another Vitis run.
* No candidate improved the baseline during this run.

## Existing Strengths

* The functional subtraction fault can be repaired successfully.
* Model-token usage is recorded.
* Only permitted source files are modified.
* Host validation and Vitis CSim work.
* PPA candidates can be statically checked before synthesis.
* Non-improving candidates are rejected.
* Failed candidates do not overwrite the working baseline.

## Existing Limitations

* Repair and PPA optimisation are separate workflows.
* The unified task manifest still depends on separate adapter configurations.
* Repair synthesis requires a manual command.
* Co-simulation is not integrated.
* Repair currently uses only one model attempt.
* Model and tool budgets are not controlled by one authoritative ledger.
* PPA generation can produce unrealistic memory transformations.
* The workflow is not yet generalised to unfamiliar hidden benchmarks.

## Regression Requirements

Future changes must continue to:

1. Detect and repair the vector-add subtraction fault.
2. Modify only permitted source files.
3. Pass host validation and Vitis CSim.
4. Pass synthesis automatically.
5. Record real model-token usage.
6. Preserve the best verified candidate when later candidates fail.
