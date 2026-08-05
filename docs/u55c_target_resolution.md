# Competition FPGA target resolution

## Decision

The final Track-A competition target is the **AMD Alveo U55C** for C/RTL co-simulation.

The supplied Track-A submission guidelines state:

- FPGA platform: Alveo U55C for co-simulation;
- software version: Vitis 2025.2;
- required stages: C simulation, synthesis and C/RTL co-simulation;
- minimum generated-hardware frequency: 100 MHz.

The exact FPGA part used by the U55C is:

```text
xcu55c-fsvh2892-2L-e
```

AMD's U55C documentation reports the following device capacities, which are used as the target-specific resource ceilings for the representative validation subset:

| Resource | U55C capacity |
|---|---:|
| LUT | 1,303,680 |
| FF/register | 2,607,360 |
| DSP | 9,024 |
| BRAM18K equivalents | 4,032 |

## Consequence for the completed reference sweep

The 36-run reference experiment targeted `xczu3eg-sfvc784-2-e`. Those results remain valid evidence for the dissertation and for evaluating the framework under the frozen reference protocol, but they must not be presented as U55C-target results.

A target change can alter timing, latency in nanoseconds, resource mapping and candidate ranking. The reference sweep therefore remains frozen, and U55C evidence is collected under a separately named validation run.

A complete repetition of all 36 runs is not required for this target check. The representative subset covers four distinct behaviours:

1. `dot_product_accumulator_overwrite`: repeatable successful latency trade-off;
2. `gemm_loop_bound_missing_k`: conservative baseline fallback;
3. `hls_eval_gesummv_accumulator_overwrite`: successful imported HLS-Eval optimisation;
4. `hls_eval_mvt_shifted_second_vector`: difficult imported case with no fully verified optimisation candidate in the reference sweep.

Each case is run once with the same model, 10 ns clock, 100 MHz minimum frequency, shared budget and validation policy as the reference experiment. The only intended experimental change is the FPGA target and its physical resource ceilings.

## Prepare the target-specific manifests

```bash
python3 scripts/prepare_u55c_validation_subset.py
```

Generated suite index:

```text
configs/tasks/u55c_validation/index.json
```

The generator creates independent manifests and target-specific `task.cfg` files. It does not edit the frozen reference manifests or benchmark files.

## Verify local U55C synthesis support without using model tokens

```bash
python3 - <<'PY'
from pathlib import Path

from agent.config import load_task
from agent.tools.synthesis import run_synthesis

manifest = Path(
    "configs/tasks/u55c_validation/"
    "dot_product_accumulator_overwrite_repair_full_agent_u55c.json"
)
task = load_task(manifest)
source = Path(task.data["artifacts"]["source"])
report = run_synthesis(task, source)

print("passed:", report["passed"])
print("failure_class:", report["failure_class"])
print("generated_tcl:", report["generated_tcl"])
print("metrics:", report["metrics"])

raise SystemExit(0 if report["passed"] else 1)
PY
```

Confirm the generated Tcl contains the intended target:

```bash
grep -R "set_part xcu55c-fsvh2892-2L-e" \
  runs/u55c_validation/dot_product_accumulator_overwrite_repair_full_agent_u55c/synthesis
```

## Dry-run and execute the representative suite

```bash
python3 scripts/run_overnight_agent_suite.py \
  --index configs/tasks/u55c_validation/index.json \
  --run-id u55c_representative_20260805 \
  --repeats 1 \
  --dry-run
```

Then run:

```bash
python3 -u scripts/run_overnight_agent_suite.py \
  --index configs/tasks/u55c_validation/index.json \
  --run-id u55c_representative_20260805 \
  --repeats 1 \
  --timeout-seconds 1800 \
  2>&1 | tee u55c_representative_20260805.log
```

Extract the final results:

```bash
python3 scripts/extract_final_results.py \
  runs/overnight_repair/u55c_representative_20260805
```

## Completion criteria

Step 4 is complete when:

- the local Vitis installation accepts `xcu55c-fsvh2892-2L-e`;
- all four runs produce a fully verified final design, either an eligible candidate or a retained verified baseline;
- synthesis and C/RTL co-simulation logs show the U55C part;
- the U55C result archive is kept separate from the frozen 36-run reference sweep.
