# ATAX diagnosis-guided optimisation

## Baseline

Candidate: `atax_candidate_3b.cpp`

- Top latency: 5936 cycles
- Interval: 5937 cycles
- Dot-loop latency: 93 cycles
- Dot-loop II: 4
- LUT: 1868
- FF: 1823
- DSP: 14
- BRAM: 0
- Estimated clock: 10.565 ns

## Diagnosis

The hierarchical analyser selected `kernel_atax_Pipeline_dot_loop`.

Source analysis identified two floating-point accumulator recurrences:

- `acc0 = acc0 + ...`
- `acc1 = acc1 + ...`

The selected transformation increased the number of independent accumulators
from two to four.

## Candidate result

Candidate: `atax_candidate_diagnosis_4acc.cpp`

- Top latency: 5176 cycles
- Interval: 5177 cycles
- Dot-loop latency: 51 cycles
- Dot-loop II: 4
- LUT: 2617
- FF: 2968
- DSP: 25
- BRAM: 0
- Estimated clock: 10.565 ns

## Interpretation

- Top latency improved by 12.8%.
- Dot-loop latency improved by 45.2%.
- The recurrence II remained 4.
- LUT increased by 40.1%.
- FF increased by 62.8%.
- DSP increased by 78.6%.
- The critical-path violation remained.

This candidate is a lower-latency but higher-area Pareto point.
