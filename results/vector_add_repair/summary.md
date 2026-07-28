# Vector-Add Repair Experiment Summary

## Research question

How effectively can a coding agent repair controlled faults in a simple AMD/Xilinx HLS vector-add design, and how does structured tool-feedback-assisted repair compare with autonomous repository inspection?

## Experimental design

Four deterministic faults were evaluated:

1. missing semicolon (`compile`)
2. subtraction instead of addition (`functional`)
3. off-by-one array indexing (`functional`)
4. incorrect top-function name (`interface`)

Two repair modes were used.

### Feedback-assisted mode

The repair model received the broken implementation, the classified failure type, and extracted Vitis HLS CSim evidence. The generated candidate was validated in an isolated copy of the benchmark using Vitis HLS CSim.

### Autonomous inspection mode

The model received only the benchmark directory and a general repair objective. Explicit fault metadata was removed. The model inspected the implementation, header, testbench, and HLS configuration, reproduced the failure using `g++`, diagnosed the defect, modified only the implementation source, and reran the host-side test. Each final candidate was then independently validated using Vitis HLS CSim through `agent/run_repair.py`.

## Results

| Fault | Intended class | Feedback-assisted repair | Autonomous diagnosis | Autonomous edit | Host validation | Independent Vitis validation |
|---|---|---:|---:|---:|---:|---:|
| Missing semicolon | Compile | Passed | Correct | One line / one hunk | Passed | Passed |
| Subtraction | Functional | Passed | Correct | One line / one hunk | Passed | Passed |
| Off-by-one indexing | Functional | Passed | Correct | One line / one hunk | Passed | Passed |
| Wrong top-function name | Interface | Passed | Correct | One line / one hunk | Passed | Passed |

### Aggregate results

- Feedback-assisted mode: **4/4 repairs passed Vitis HLS CSim**.
- Autonomous inspection mode: **4/4 diagnoses were correct**.
- Autonomous inspection mode: **4/4 repairs passed the host-side C++ test**.
- Autonomous inspection mode: **4/4 repaired designs passed independent Vitis HLS CSim validation**.
- Every successful repair was confined to a single local code hunk.
- No header, testbench, or configuration files were modified.

## Autonomous token usage

| Fault | Tokens used |
|---|---:|
| Subtraction | 36,322 |
| Off-by-one indexing | 34,720 |
| Missing semicolon | 9,952 |
| Wrong top-function name | 16,458 |

The first two autonomous runs attempted to invoke the Vitis toolchain from inside the Codex sandbox. Vitis repeatedly stalled during XCD startup because socket creation was not permitted. Later prompts explicitly used host-side `g++` validation and delegated trusted Vitis validation to the outer workflow, substantially reducing unnecessary exploration.

## Behavioural observations

- The agent inspected the source, interface, testbench, and configuration before editing.
- It inferred intended behaviour from the testbench without explicit fault labels.
- It reproduced failures before modifying the implementation.
- It made minimal, source-only repairs.
- It preserved the top-function signature and testbench compatibility.
- It avoided unnecessary edits when presented with an already-correct implementation.
- Tool-use autonomy increased token cost substantially when the HLS launcher was unavailable inside the sandbox.

## Interpretation

The experiment demonstrates two successful repair strategies on controlled HLS faults. Structured Vitis feedback provides a compact and reliable repair path, while autonomous repository inspection can independently infer the intended behaviour and identify the defect. However, autonomous tool exploration is more expensive and can be disrupted by sandbox restrictions. A practical architecture is therefore to allow the model to inspect and repair using lightweight host simulation, while an outer controller performs authoritative Vitis HLS validation.

## Scope and limitations

These results are based on one simple vector-add design and four single-line faults. They do not yet demonstrate generalisation to larger HLS benchmarks, multi-fault designs, synthesis failures, or quality-of-result optimisation. Repeated trials and additional models are required before making broader claims about reliability.
