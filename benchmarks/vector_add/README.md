# Controlled vector-add repair experiment

This benchmark is the first narrow repair milestone for the project. It tests whether the existing agent can diagnose and repair controlled faults using Vitis HLS C-simulation feedback before any optimisation work is attempted.

## Fault classes

- `syntax_missing_semicolon`: compiler-visible syntax failure.
- `functional_subtraction`: compiles but violates the addition specification.
- `indexing_off_by_one`: compiles but produces incorrect element-wise results.
- `interface_wrong_top_name`: breaks compatibility with the declared interface and testbench.

## Prepare the experiment

```bash
python3 benchmarks/vector_add/create_faults.py
```

First verify the golden implementation:

```bash
python3 agent/run_repair.py benchmarks/vector_add/golden
```

It should pass and report that no repair is required.

Then run one controlled repair at a time:

```bash
python3 agent/run_repair.py \
  benchmarks/vector_add/faults/functional_subtraction \
  --generate-repair \
  --validate
```

Repeat the command for each generated fault directory. Keep the model fixed for the first comparison.

## Record for each run

Record the fault class, model, extracted evidence, whether the diagnosis matched the injected fault, validation pass/fail, edit size, unrelated edits, and notable agent behaviour. The generated `runs/` directory already preserves the prompt, Codex log, candidate source, validation log, and JSON result.

## Scope boundary

This experiment measures functional repair only. Synthesis-quality optimisation, Pareto search, multiple HLS benchmarks, and model-ranking experiments should follow only after this repair suite is reproducible.
