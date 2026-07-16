# Repair Experiment Results

| Experiment | Bug type | Initial result | Codex attempts | Final result | Key observation |
|---|---|---|---:|---|---|
| Adder subtraction | Wrong operator | CSim failed | 1 | CSim + synthesis passed | Simple functional bug repaired directly |
| Adder output reference | Pass-by-value interface bug | CSim returned zero | 1 | CSim + synthesis passed | Codex identified data-flow/interface issue |
| Vector addition | Off-by-one loop bound | Final element incorrect | 1 | CSim + synthesis passed | Repair transferred to an array-based pipelined kernel |
| Dot product | Accumulator overwritten | Returned only final product | 1 | CSim + synthesis passed | Codex repaired loop-carried accumulation |

## Common configuration

- Vitis HLS 2025.2
- FPGA part: `xczu3eg-sfvc784-2-e`
- Target clock: `10 ns`
