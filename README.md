# LLM4HLS Agent Project

This repository contains experiments for generating, debugging and
optimising Xilinx HLS code using large language models.

## Current objectives

1. Establish a reproducible Vitis HLS workflow.
2. Test prompt-based repair of deliberately faulty HLS implementations.
3. Reproduce an existing LLM-based HLS generation workflow.
4. Extend the workflow using structured compiler, simulation and
   synthesis feedback.

## Repository structure

- `benchmarks/`: HLS source code and testbenches
- `runs/`: ignored, complete per-run bundles (logs, intermediate output, and workspaces)
- `results/`: tracked, compact derived summaries and comparison tables
- `evidence/`: tracked, deliberately selected artifacts that support a reported result
- `prompts/`: exact prompts supplied to language models
- `scripts/`: automation, grouped by workflow:
  - `analysis/`: HLS evidence extraction and diagnosis
  - `experiments/`: repair experiment runners and suites
  - `ppa/`: PPA candidate generation, validation, synthesis, and refinement
  - `ablations/`: repeated-run ablations and result comparisons
  - `track_a/`: Track A task preparation and autonomous runs
  - `setup/`: benchmark and configuration setup
- `notes/`: environment details, decisions and laboratory logs

## First experiment

The first experiment uses a simple integer adder to validate the complete
generate-test-feedback-repair workflow.

## Repository hygiene

### What belongs in Git

Keep the source of truth needed to run or reproduce an experiment:

- `agent/`, `scripts/`, and `configs/`
- benchmark source, headers, testbenches, task configuration, and fault
  descriptions
- prompts, concise documentation, and selected result summaries needed to
  support reported conclusions

`runs/` is the operational record for an execution and is intentionally ignored
by Git. `results/` and `evidence/` are the small, reviewed record retained in
Git; neither is a runtime dependency. See `docs/output-storage.md` for the
promotion rules.

### Do not add generated local artifacts

The following are reproducible by running the workflow and should not be
committed in new changes:

- Vitis workspaces such as `benchmarks/real/*/work/` (build metadata,
  dependency files, and simulation scripts)
- compiled `host_test` binaries
- Vitis launcher metadata such as `*.app`
- autonomous-run and validation logs

The ignore rules now cover `host_test` and `*.app`; the existing rules already
cover `work/`, `*.log`, and `runs/`.

### Audit recorded 2026-08-02

The repository currently tracks generated artifacts that are outside the
execution path: 26 files under `benchmarks/real/f_class/work/` and
`benchmarks/real/s_class/work/`, four `host_test` ELF binaries, four
`experiments/autonomous_vector_add_*/codex_autonomous.log` files, and
`benchmarks/vector_add/ppa_baseline/vector_add_hls/hls.app`. These do not need
to be pushed going forward. They remain in the current checkout so this audit
does not discard existing material.

To remove them from a future commit while retaining local copies, remove them
from the Git index with `git rm --cached <paths>`, review the staged deletion,
and commit it. Keep the tracked synthesis reports, analysis JSON, and selected
CSV/summary files only when they are evidence for results you intend to retain.
