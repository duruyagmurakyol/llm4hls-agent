# Track A Agent Acceptance Contract

## Purpose

This document defines the minimum behaviour required for the LLM4HLS agent to be considered complete and ready for FPT Track A evaluation.

The acceptance contract is intended to prevent the project from becoming a collection of disconnected repair and optimisation scripts. The final system must operate as one autonomous, correctness-first and budget-aware workflow.

The agent should be executable through one public command:

```bash
python3 scripts/run_agent.py <task>
```

## Input Contract

The agent receives an unfamiliar HLS task directory or task manifest containing some or all of the following:

* one or more HLS C or C++ design source files;
* header files and include directories;
* one or more testbench files;
* a build configuration such as `task.cfg` or TCL;
* the declared top function;
* an optional natural-language specification;
* editable and protected file definitions;
* the target FPGA platform or part;
* the target clock period or minimum frequency;
* optional resource constraints;
* numerical tolerance requirements;
* model, token and tool-call budgets;
* an output directory.

The agent must not require benchmark-specific code paths or hard-coded knowledge of task names, array sizes or loop labels.

## Required Agent Behaviour

### 1. Task Discovery

The agent must:

* locate the design source files;
* locate testbench and header files;
* determine the top function;
* identify the build configuration;
* preserve include paths and compiler options;
* identify editable and protected files;
* determine the FPGA target and clock constraint;
* report an onboarding failure if required information cannot be determined safely.

Task discovery must not consume model or synthesis budget when the task configuration is invalid.

### 2. Initial Validation

The agent must determine the earliest failing stage of the supplied design.

The validation sequence should include, where required:

1. static source validation;
2. host compilation and execution;
3. Vitis CSim;
4. Vitis synthesis;
5. Vitis co-simulation.

The agent must distinguish between at least:

* compilation or syntax failure;
* interface or linkage failure;
* functional mismatch;
* timeout;
* synthesis failure;
* co-simulation failure;
* unknown failure.

### 3. Correctness Repair

If the design is not correct, the agent must enter a repair loop.

The repair loop must:

1. classify the observed failure;
2. extract concise evidence from the relevant logs;
3. generate a constrained repair prompt;
4. ask the model to modify only permitted source files;
5. validate the model output before applying it;
6. rerun the relevant validation stages;
7. feed new failure evidence into the next attempt;
8. stop when the design passes or the budget is exhausted.

The repair process must preserve:

* the declared top-function interface;
* protected source files;
* testbench behaviour;
* build configuration;
* task constraints.

The agent must not modify the testbench to make an incorrect design pass.

### 4. Verified Baseline Establishment

Once a design passes all required correctness checks, the agent must establish an immutable verified baseline.

The baseline record must include:

* source hash;
* CSim result;
* synthesis result;
* co-simulation result;
* estimated clock period;
* estimated frequency;
* latency in cycles;
* initiation interval;
* LUT usage;
* FF usage;
* DSP usage;
* BRAM usage.

The verified baseline must never be overwritten by a later failed optimisation candidate.

### 5. PPA Optimisation

The agent may begin PPA optimisation only after establishing a verified correct baseline.

For each optimisation candidate, the agent must:

1. identify a hardware bottleneck or optimisation opportunity;
2. record an explicit optimisation hypothesis;
3. generate a constrained candidate;
4. run cheap static validation;
5. reject duplicate or invalid candidates;
6. run CSim;
7. run synthesis;
8. run co-simulation where required;
9. extract hardware metrics;
10. compare the candidate against the verified baseline and current Pareto archive;
11. record an acceptance or rejection verdict.

Optimisation decisions should use evidence such as:

* loop initiation interval;
* latency;
* memory-port bottlenecks;
* recurrence constraints;
* resource usage;
* timing estimates;
* synthesis failures;
* co-simulation failures.

### 6. Frequency and PPA Requirements

The final candidate must satisfy the configured clock or frequency requirement.

For a 100 MHz target:

```text
maximum estimated clock period = 10 ns
```

The agent must evaluate actual estimated latency using:

```text
latency time = latency cycles × estimated clock period
```

The agent must not treat fewer cycles as an improvement if the clock period worsens enough to increase total latency.

Candidate evaluation must consider:

* functional correctness;
* synthesis success;
* co-simulation success;
* target frequency;
* latency;
* throughput or initiation interval;
* LUT usage;
* FF usage;
* DSP usage;
* BRAM usage;
* task-specific resource limits.

### 7. Budget Control

All expensive actions must be controlled by one authoritative budget ledger.

The ledger must track:

* model calls;
* input tokens;
* output tokens;
* total tokens;
* CSim calls;
* synthesis calls;
* co-simulation calls;
* optimisation or repair iterations;
* any organiser-defined unified credit budget.

Failed and timed-out calls must still be counted when the external tool or model was invoked.

The agent must reserve sufficient remaining budget to verify the final selected candidate.

The agent must stop cleanly when:

* the budget is exhausted;
* no valid new strategy remains;
* repeated candidates are produced;
* the target improvement is achieved;
* further action would risk losing final verification capability.

### 8. Candidate Preservation

The agent must maintain:

* the original source;
* the latest candidate;
* the best verified correct candidate;
* the best PPA candidate;
* the Pareto archive;
* rejected-candidate history.

A failed later candidate must never replace the best verified candidate.

### 9. Reproducibility and Provenance

For every model or tool action, the agent must record sufficient evidence to reproduce and audit the result.

This should include:

* task identifier;
* timestamp;
* model name;
* prompt;
* raw model response;
* input and output tokens;
* model latency;
* candidate source;
* source diff;
* candidate hash;
* generated command;
* working directory;
* tool return code;
* timeout status;
* tool duration;
* log path;
* extracted metrics;
* acceptance or rejection verdict.

Each synthesis or simulation report must be traceable to the exact source candidate that produced it.

## Required Final Output

A successful run must produce:

* the final selected HLS source;
* the verified source hash;
* CSim status;
* synthesis status;
* co-simulation status;
* estimated clock period;
* estimated frequency;
* latency cycles;
* estimated latency time;
* initiation interval;
* LUT usage;
* FF usage;
* DSP usage;
* BRAM usage;
* model and token usage;
* tool-call usage;
* remaining budget;
* candidate history;
* final termination reason;
* machine-readable result files.

Suggested output files include:

```text
unified_agent_result.json
budget_summary.json
candidate_history.json
final_source.cpp
final_validation.json
final_synthesis_metrics.json
```

## Failure Behaviour

If the task cannot be completed, the agent must still:

* preserve the best verified candidate, if one exists;
* return a non-ambiguous failure status;
* record the stage at which the run stopped;
* record the remaining and consumed budget;
* retain all evidence required to diagnose the failure;
* avoid modifying protected or original source files.

## Acceptance Scenarios

The implementation satisfies this contract only when the same public command can process the following cases.

### Scenario A — Broken Functional Design

Input:

* a vector-add implementation using subtraction instead of addition.

Expected behaviour:

* detect the functional failure;
* repair only the editable source;
* pass host validation;
* pass CSim;
* pass synthesis;
* pass co-simulation;
* record tokens and tool calls;
* return the repaired source.

### Scenario B — Synthesis Failure

Input:

* a design that passes host execution or CSim but fails HLS synthesis.

Expected behaviour:

* identify synthesis as the failing stage;
* extract relevant Vitis evidence;
* repair the synthesizability issue;
* rerun validation;
* establish a verified baseline.

### Scenario C — Correct but Inefficient Design

Input:

* a functionally correct design with an obvious PPA bottleneck.

Expected behaviour:

* establish the original design as the verified baseline;
* diagnose the hardware bottleneck;
* generate one or more constrained candidates;
* preserve correctness;
* improve the configured PPA objective;
* return the strongest verified candidate.

### Scenario D — Combined Repair and Optimisation

Input:

* a design that is initially functionally incorrect and inefficient.

Expected behaviour:

* repair the design;
* establish the repaired design as the baseline;
* continue into PPA optimisation automatically;
* return the best verified final candidate in one invocation.

### Scenario E — Budget Exhaustion

Input:

* a task where the model repeatedly generates invalid or unhelpful candidates.

Expected behaviour:

* enforce all budgets;
* stop before exceeding the limits;
* preserve the best verified candidate;
* produce a structured budget-exhaustion termination reason.

### Scenario F — Unseen Benchmark

Input:

* a benchmark not referenced by name in the agent source code.

Expected behaviour:

* discover the task structure;
* perform the appropriate validation, repair and optimisation stages;
* avoid benchmark-specific assumptions;
* produce a complete result record.

## Definition of Completion

The Track A agent is considered minimally complete when:

```bash
python3 scripts/run_agent.py <unknown-task>
```

can, without manual file movement or manual Vitis commands:

1. discover the task;
2. identify the initial failure stage;
3. repair the design iteratively when required;
4. pass CSim;
5. pass synthesis;
6. pass co-simulation;
7. establish a verified baseline;
8. optimise PPA when budget remains;
9. satisfy the target frequency;
10. remain within model, token and tool budgets;
11. preserve the best verified candidate;
12. produce complete machine-readable evidence;
13. terminate with a clear reason.

Any workflow that requires selecting separate repair or optimisation scripts manually does not yet satisfy this contract.
