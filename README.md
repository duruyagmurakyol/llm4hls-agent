# LLM4HLS Agent

A budgeted autonomous agent for repairing incorrect AMD/Xilinx HLS designs and performing synthesis-feedback-guided PPA optimisation.

## Workflow

```text
task manifest
    |
    v
initial validation
    |-- failure --> repair adapter --> correctness validation
    |                                   |
    `-----------------------------------'
                    |
                    v
             baseline synthesis
                    |
                    v
       diagnosis and PPA iteration
                    |
                    v
         Pareto and budget decision
```

Correctness is established before synthesis-based optimisation. Candidate designs are statically checked and C-simulated before consuming a synthesis call.

## Quick start

Validate the Python integration without invoking Vitis:

```bash
python3 -m pytest tests/test_unified_config.py tests/test_unified_state.py
```

Run the controlled vector-add functional repair:

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_repair.json
```

Run or inspect the vector-add PPA task:

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_track_a.json --status-only
python3 scripts/run_agent.py configs/tasks/vector_add_track_a.json
```

The public entry point is `scripts/run_agent.py`. It loads a unified task manifest and routes to the currently proven repair or PPA adapter.

## Active architecture

```text
agent/
  config.py                 task validation
  state.py                  shared budgets and result records
  controller.py             unified routing and result writing
  providers/                model-provider clients
  analysis/                 Vitis evidence and bottleneck diagnosis

scripts/
  run_agent.py              public CLI
  run_api_experiment.py     current direct-API repair adapter
  run_track_a_agent.py      current budgeted PPA adapter
  run_ppa_agent_iteration.py
  run_ppa_candidate_csim.py
  run_ppa_candidate_synthesis.py
  evaluate_ppa_experiment.py

configs/tasks/              unified task manifests
benchmarks/                 HLS source, headers, tests and build files
experiments/                generated task trajectories and summaries
results/                    selected repair and analysis records
```

The existing experiment scripts remain temporarily as compatibility adapters. Reusable logic will be extracted only after the unified command reproduces the known repair and PPA results.

## Current evidence

The controlled vector-add repair suite previously repaired functional, indexing, interface and syntax faults while preserving protected files.

The current vector-add PPA experiment records:

| Design | Latency (cycles) | LUT | FF | DSP |
|---|---:|---:|---:|---:|
| Baseline | 1029 | 320 | 321 | 2 |
| Initial parallel candidate | 517 | 11296 | 17541 | 4 |
| Refined banked candidate | 517 | 518 | 548 | 4 |

## Refactor policy

1. Preserve working repair and Vitis flows.
2. Expose them through one task manifest and one CLI.
3. Validate known results through the unified controller.
4. Extract shared validation, synthesis, repair and optimisation modules.
5. Archive obsolete wrappers and generated artefacts only after regression checks pass.

## Requirements

- Python 3.10 or later
- AMD Vitis HLS 2025.2 for CSim and synthesis tasks
- `SILICONFLOW_API_KEY` for direct model calls
- `pytest` for lightweight integration tests
