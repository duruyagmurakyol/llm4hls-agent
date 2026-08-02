# Script catalogue

`scripts/` contains command-line entry points only. Reusable implementation code belongs in `agent/`.

## Folder ownership

| Folder | Purpose | Add a script here when it… |
| --- | --- | --- |
| `experiments/` | Single-run and suite orchestration | launches or aggregates configured repair experiments |
| `ablations/` | Controlled repeated studies | compares modes, benchmarks, models, or iteration budgets |
| `analysis/` | Post-processing | reads existing evidence/results and produces diagnoses or comparisons |
| `setup/` | Deterministic data/config generation | creates benchmark faults, configs, or manifests |
| `providers/` | Provider administration | inspects model catalogues or provider capabilities |
| `maintenance/` | One-off repository migrations | patches or migrates existing project files |

## Experiments

| Script | Responsibility |
| --- | --- |
| `experiments/run_experiment.py` | Autonomous Codex repair with a mutable workspace and independent validation |
| `experiments/run_structured_experiment.py` | Codex repair using bounded, runner-supplied failure evidence and context |
| `experiments/run_api_experiment.py` | One-shot direct SiliconFlow API repair |
| `experiments/run_iterative_api_experiment.py` | Bounded direct-API repair loop with validation feedback between iterations |
| `experiments/run_suite.py` | Dispatch all JSON configs in a directory and aggregate their results |

## Ablations

| Script | Responsibility |
| --- | --- |
| `ablations/run_bicg_staged_ablation.py` | One-shot versus iterative staged BICG comparison |
| `ablations/run_bicg_staged_repeated_ablation.py` | Repeated staged BICG comparison with variance and anomaly reporting |
| `ablations/run_atax_staged_repeated_ablation.py` | Repeated staged ATAX comparison |
| `ablations/run_staged_model_repeated.py` | Repeated staged comparison for a configured model across BICG and ATAX |

## Analysis

| Script | Responsibility |
| --- | --- |
| `analysis/compare_atax_bicg_staged_results.py` | Compare the latest repeated ATAX and BICG results |
| `analysis/compare_staged_models.py` | Compare the latest staged results across configured models |
| `analysis/diagnose_hls_evidence.py` | Convert generic HLS evidence into a structured bottleneck diagnosis |

## Setup and provider tools

| Script | Responsibility |
| --- | --- |
| `setup/setup_bicg_repair_suite.py` | Generate the basic BICG fault suite and configs from the golden design |
| `setup/setup_atax_staged_ablation.py` | Generate staged ATAX faults and iterative configs |
| `setup/setup_bicg_staged_ablation.py` | Generate staged BICG faults and iterative configs |
| `setup/setup_staged_model_configs.py` | Clone verified staged configs for another SiliconFlow model |
| `providers/list_siliconflow_models.py` | List competition-relevant models exposed by SiliconFlow |

## Maintenance

`maintenance/harden_api_response_parsing.py` is a historical migration utility. It is retained for reproducibility, but it is not part of the normal experiment workflow.

## Adding a script

1. Put reusable logic in `agent/` first.
2. Keep the CLI small and configuration-driven.
3. Resolve the repository root with `Path(__file__).resolve().parents[2]` from these category folders.
4. Write generated data beneath `results/`, or deterministic generated inputs beneath `benchmarks/`/`configs/`.
5. Add the command and output location to this catalogue or `docs/running-experiments.md`.
