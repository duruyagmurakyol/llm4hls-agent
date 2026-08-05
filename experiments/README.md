# `experiments/`

This directory stores generated run state and evidence. Files here are outputs of the agent, not the implementation and not the original benchmark source.

## Typical structure

```text
experiments/
├── onboarding/
│   └── <benchmark>/
│       ├── task.json
│       ├── optimisation.json
│       ├── onboarding_report.json
│       ├── autonomous_ppa/
│       └── agent_result/
├── track_a/
│   └── <task-id>/
└── <named-experiment>/
```

Exact subdirectories vary by task and workflow.

## Onboarding outputs

For a benchmark-directory input, onboarding creates:

- `task.json`: generated unified manifest;
- `optimisation.json`: generated PPA configuration;
- `onboarding_report.json`: source selection and discovery evidence.

These files describe what the agent inferred from the benchmark. Review them when adding a new layout.

## Optimisation outputs

A PPA output directory may contain:

```text
baseline_run/
baseline_synthesis.json
baseline_hierarchical_diagnosis.json
baseline_source_target.json
baseline_source_cause.json
candidate_001_prompt.txt
candidate_001.cpp
candidate_001_generation.json
candidate_001_static_validation.json
candidate_001_duplicate_check.json
candidate_001_csim_validation.json
candidate_001_synthesis.json
candidate_001_synthesis/
optimisation_state.json
```

Names can evolve, but each stage should leave enough machine-readable evidence to explain the next decision.

## Unified result

The controller writes:

```text
agent_result/unified_agent_result.json
```

This file summarises:

- task identifier;
- success status;
- termination reason;
- ordered trajectory events;
- output location.

It is the top-level record for a run, while stage reports contain detailed evidence.

## Resumability

The optimisation runner examines existing artefacts and continues from the next incomplete stage. This prevents repeated model, CSim or synthesis calls after an interrupted run.

Before rerunning, prefer:

```bash
python3 scripts/run_agent.py <task> --status-only
```

Do not delete a run merely because a candidate was rejected; rejection reports are valid experimental evidence.

## When it is safe to delete

It is generally safe to remove:

- an obviously incomplete development run that will not be analysed;
- temporary generated projects after their reports have been recorded;
- duplicate onboarding outputs created during layout debugging.

Be cautious before deleting:

- model prompts and candidate sources;
- baseline and candidate metric JSON;
- failure logs that explain agent behaviour;
- accepted or Pareto-relevant candidates;
- outputs referenced by a dissertation, report or submission.

## What should be committed

Commit selected experiment evidence only when it is intentionally part of the research record. Large Vitis project directories, generated RTL and tool caches should normally remain ignored.

A retained experiment should ideally include:

- exact task and optimisation config;
- baseline metrics;
- candidate source;
- validation and synthesis reports;
- final verdict;
- model name and token metadata;
- enough logs to reproduce a failure.

## Difference from `results/`

- `experiments/` contains working run trajectories, including failures and rejected candidates.
- `results/` contains curated evidence selected for comparison, reporting or submission.

Do not manually edit generated metrics to improve presentation. Derive curated tables from the original evidence instead.
