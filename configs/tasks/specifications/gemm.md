# GEMM

Compute:

C = beta * C + alpha * A * B

Dimensions:

- C: 20 × 25
- A: 20 × 30
- B: 30 × 25

Preserve the exact `kernel_gemm` interface and numerical behaviour checked by
the supplied self-checking testbench.
