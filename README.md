# LLM4HLS Agent

A reproducible research repository for generating, repairing, evaluating, and optimising AMD/Xilinx HLS designs with language-model agents.

## Research workflow

The repository supports four connected activities:

1. Define a benchmark and intentionally faulty variants.
2. Run a repair strategy against a JSON experiment configuration.
3. Validate the candidate with host tests and an independent HLS flow.
4. Aggregate repeated runs to study correctness, token use, latency, and repair behaviour.

## Repository layout

```text
agent/          Reusable repair, provider, and HLS-analysis code
benchmarks/     Golden HLS designs, fault variants, tests, and task metadata
configs/        Versioned experiment and model configurations
docs/           Repository and experiment documentation
notes/          Research notes and observations
prompts/        Reusable prompt material
results/        Generated experiment, suite, ablation, and comparison artefacts
scripts/        Thin command-line entry points grouped by responsibility
```

See [docs/repository-layout.md](docs/repository-layout.md) for ownership rules and [scripts/README.md](scripts/README.md) for the complete script catalogue.

## Common commands

Run one direct API repair:

```bash
export SILICONFLOW_API_KEY="..."
python3 scripts/experiments/run_api_experiment.py \
  configs/vector_add_api_qwen35/functional.json
```

Run a bounded iterative repair:

```bash
python3 scripts/experiments/run_iterative_api_experiment.py \
  configs/bicg_iterative_qwen35/staged_compile_then_functional.json \
  --max-iterations 3
```

Run every configuration in a suite:

```bash
python3 scripts/experiments/run_suite.py configs/vector_add_api_qwen35 \
  --continue-on-failure
```

Diagnose synthesised HLS evidence:

```bash
python3 scripts/analysis/diagnose_hls_evidence.py evidence.json
```

More examples and the expected output structure are in [docs/running-experiments.md](docs/running-experiments.md).

## Conventions

- Keep reusable logic in `agent/`; scripts should mainly parse arguments and orchestrate workflows.
- Keep benchmark-specific source, tests, and fault metadata under the relevant benchmark.
- Keep experiment parameters in JSON under `configs/`, rather than hard-coding new parameters into runners.
- Write generated artefacts only under `results/`; do not edit historical result directories after a run.
- Put new command-line tools in the matching `scripts/` subdirectory documented in `scripts/README.md`.
