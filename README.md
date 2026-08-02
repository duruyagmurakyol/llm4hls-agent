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
- `experiments/`: individual experimental runs and metadata
- `prompts/`: exact prompts supplied to language models
- `results/`: selected simulation and synthesis results
- `scripts/`: experiment automation and report parsing
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

The `results/`, `logs/`, and `experiments/` directories are research records,
not runtime dependencies. Keep a small, intentional set of summaries there;
archive detailed run outputs elsewhere when they are not required for a paper
or a reproducibility release.

### Do not add generated local artifacts

The following are reproducible by running the workflow and should not be
committed in new changes:

- Vitis workspaces such as `benchmarks/real/*/work/` (build metadata,
  dependency files, and simulation scripts)
- compiled `host_test` binaries
- Vitis launcher metadata such as `*.app`
- autonomous-run and validation logs

The ignore rules now cover `host_test` and `*.app`; the existing rules already
cover `work/`, `*.log`, and `results/experiments/`.

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
