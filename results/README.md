# `results/`

This directory contains curated experimental evidence selected from generated runs for analysis, comparison, reporting or submission.

It is different from `experiments/`:

- `experiments/` is the working history of agent runs, including rejected candidates and debugging evidence.
- `results/` is a deliberate, smaller collection of evidence that supports conclusions.

## What belongs here

Examples include:

- baseline-versus-candidate metric summaries;
- accepted or Pareto-relevant candidate records;
- benchmark comparison tables;
- aggregated success and failure counts;
- selected trajectory summaries;
- data used directly in a dissertation figure or competition report.

## Required provenance

Every retained result should be traceable to:

- benchmark and task identifier;
- exact source revision or commit;
- task and optimisation configuration;
- Vitis version, part and clock target;
- model provider and model name;
- baseline and candidate source paths;
- original reports under `experiments/`;
- acceptance or rejection rule.

Do not copy a number into a summary without preserving where it came from.

## Recommended format

Prefer machine-readable JSON or CSV for data and Markdown for explanations.

Example:

```text
results/
└── bicg/
    ├── summary.md
    ├── metrics.json
    └── provenance.json
```

A summary should distinguish clearly between:

- functional correctness;
- initiation interval;
- top-level latency and interval;
- estimated clock period;
- LUT, FF, DSP and BRAM usage;
- whether timing constraints were met;
- final agent verdict.

## Interpretation rules

- Do not claim an optimisation from loop II alone.
- Compare baseline and candidate under identical part, clock and tool settings.
- Report timing regressions even when latency improves.
- Keep rejected candidates when they demonstrate a meaningful agent behaviour or failure mode.
- Separate measured synthesis evidence from hypotheses about why a result occurred.
- Do not manually alter generated metrics.

## Reproducibility

A result intended for formal use should be reproducible from a documented command, for example:

```bash
python3 scripts/run_agent.py configs/tasks/atax_track_a.json
```

Where automatic onboarding was used, retain or copy the generated task and optimisation config alongside the selected evidence.
