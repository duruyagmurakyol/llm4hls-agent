# `tests/`

This directory contains fast regression tests for the unified agent architecture and generic safety rules.

The tests are intentionally lightweight: they validate Python logic, manifests and generated-command construction without requiring a full Vitis synthesis run on every test invocation.

## Running the suite

```bash
python -m pytest
```

Also check import and syntax integrity:

```bash
python -m compileall -q agent scripts tests
```

## Current test areas

### Unified configuration

Tests verify that task manifests load correctly, required fields are present, adapter kinds are valid and path/configuration errors fail clearly.

### Shared state

Tests cover metric and result records used across repair and optimisation.

### Architecture

Structural tests ensure that:

- core packages import cleanly;
- obsolete experiment-specific scripts remain removed;
- task manifests do not embed shell orchestration;
- strategy guidance is benchmark-independent;
- benchmark onboarding discovers expected metadata;
- candidate status handling is resumable.

### Static HLS safety checks

Regression tests cover rules such as:

- `DATAFLOW` and incompatible loop-level pipelining in the same region;
- full unrolling combined with pipelining on the same loop;
- complete partitioning of public interface arrays;
- allowed partial unroll and pipeline combinations.

### Evaluation

Tests cover duplicate-source normalisation, Pareto dominance and terminal synthesis outcomes such as timeouts.

## What should have a test

Add a regression test whenever changing:

- onboarding of a build-description format;
- path resolution;
- generated TCL construction;
- static validation rules;
- failure classification;
- budget or resumability logic;
- metric extraction or candidate verdicts;
- public task-manifest structure.

A genericity bug discovered on one benchmark should become a small benchmark-independent test.

## Testing `task.cfg` support

Tests for `task.cfg` should check both levels:

1. onboarding extracts the source, top, testbench, part and clock;
2. synthesis utilities convert the config into correct internal TCL commands, including absolute include paths and testbench flags.

The first level alone does not prove that candidate CSim and synthesis will work.

## Vitis integration checks

Full integration runs are performed manually on the configured Linux/Vitis machine:

```bash
python3 scripts/run_agent.py benchmarks/vector_add --max-agent-steps 1
python3 scripts/run_agent.py benchmarks/hls_eval/atax --max-agent-steps 1
python3 scripts/run_agent.py benchmarks/bicg/golden --max-agent-steps 1
```

Record successful integration evidence under `experiments/`. Do not make the normal unit suite depend on API keys or a Vitis installation.

## Test-writing rules

- Keep tests deterministic.
- Avoid live model API calls.
- Avoid expensive synthesis in unit tests.
- Use temporary directories for generated files.
- Assert behaviour and evidence, not incidental log formatting.
- Use minimal synthetic sources to isolate one rule.
- Do not encode benchmark-specific optimisation answers as core expectations.
