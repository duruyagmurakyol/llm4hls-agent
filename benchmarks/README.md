# `benchmarks/`

This directory contains HLS designs used as inputs to the unified agent. A benchmark is data: source code, headers, testbench and build description. It must not contain a benchmark-specific agent implementation.

## Supported benchmark layouts

The agent currently supports two build-description styles.

### TCL-based layout

```text
benchmarks/example/
├── src/
│   ├── kernel.cpp
│   └── kernel.h
├── testbench/
│   └── kernel_test.cpp
└── scripts/
    └── run_hls.tcl
```

The selected TCL should identify:

- the top function with `set_top`;
- design files with `add_files`;
- testbench files with `add_files -tb`;
- a solution;
- the FPGA part;
- the target clock;
- a synthesis flow.

Example:

```tcl
open_project -reset kernel_project
set_top kernel
add_files ../src/kernel.cpp
add_files ../src/kernel.h
add_files -tb ../testbench/kernel_test.cpp
open_solution -reset solution1
set_part xczu3eg-sfvc784-2-e
create_clock -period 10 -name default
csim_design
csynth_design
```

### `task.cfg`-based layout

```text
benchmarks/example/golden/
├── src/
│   ├── kernel.cpp
│   └── kernel.h
├── testbench/
│   └── kernel_test.cpp
└── task.cfg
```

Example:

```ini
[hls]
flow_target=vivado
syn.file=src/kernel.cpp
syn.cflags=-Isrc
syn.top=kernel
tb.file=testbench/kernel_test.cpp
tb.cflags=-Isrc
part=xczu3eg-sfvc784-2-e
clock=10ns
```

Paths in `task.cfg` are resolved relative to the directory containing the config.

## Golden designs and faults

Some benchmark families contain both a correct design and intentionally faulty variants:

```text
benchmark/
├── golden/
└── faults/
    ├── syntax_missing_semicolon/
    ├── functional_wrong_operator/
    └── interface_wrong_top_name/
```

- `golden/` is the correct reference and can be used for PPA optimisation.
- `faults/` contains repair tasks used to test diagnosis and correction.

A faulty design should change the source or build description only as required by the fault. The testbench should continue to express the intended behaviour.

## Running benchmarks

Onboard and run a benchmark directory:

```bash
python3 scripts/run_agent.py benchmarks/vector_add
python3 scripts/run_agent.py benchmarks/hls_eval/atax
python3 scripts/run_agent.py benchmarks/bicg/golden
```

Generate configurations only:

```bash
python3 scripts/run_agent.py benchmarks/bicg/golden --onboard-only
```

Generated files are written under `experiments/onboarding/`; the benchmark directory itself should remain unchanged.

## Benchmark requirements

A benchmark intended for automatic onboarding must provide enough information to determine exactly one of each:

- top-level function;
- primary design source;
- one or more testbench sources;
- FPGA part;
- clock period;
- build description.

The design and testbench must be inside the repository. Ambiguous or missing metadata should cause onboarding to fail clearly rather than guess.

## Source requirements

- Use synthesizable C or C++ accepted by the installed Vitis HLS version.
- Preserve a stable top-level function signature.
- Keep constants and dimensions in headers where practical.
- Avoid hidden dependencies outside the benchmark package.
- Include all required include paths in TCL or `task.cfg`.
- Use deterministic tests.

## Testbench requirements

The testbench is the correctness oracle for CSim. It should:

- call the configured top function;
- cover all outputs affected by the design;
- fail with a non-zero exit code on mismatch;
- print a clear success message when all checks pass;
- avoid changing expected outputs to accommodate generated candidates;
- remain computationally small enough for repeated agent runs.

## Adding a new benchmark

1. Create a self-contained folder.
2. Add the source and any headers.
3. Add a deterministic testbench.
4. Add either a valid synthesis TCL or `task.cfg`.
5. Run the original design directly with Vitis.
6. Run onboarding only.
7. Inspect the generated `task.json`, `optimisation.json` and onboarding report.
8. Run one agent step before allowing a larger budget.

Example validation:

```bash
python3 scripts/run_agent.py benchmarks/example --onboard-only
python3 scripts/run_agent.py benchmarks/example --max-agent-steps 1
```

## Do not store here

Do not commit the following inside benchmark folders:

- generated Vitis projects;
- `.Xil` directories;
- candidate sources produced by the agent;
- model responses;
- synthesis logs and XML copied from experiments;
- accepted-result summaries.

Those belong under `experiments/` or `results/`.
