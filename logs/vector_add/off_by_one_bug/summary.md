# Vector Addition Off-by-One Repair Experiment

## Objective

Test whether Codex could repair an off-by-one loop error in a Xilinx HLS vector addition kernel.

## Fault Introduced

The loop bound was changed from:

```cpp
i < VECTOR_SIZE
```

to:

```cpp
i < VECTOR_SIZE - 1
```

This caused the final output element to remain unchanged.

## Failure Result

C simulation failed at the final array index because the last vector element was not processed.

## Repair Result

Codex restored the correct loop bound.

The repaired implementation completed successfully:

* C simulation: Pass
* C synthesis: Pass
* RTL generation: Pass
* Initiation interval: `1`
* Pipeline depth: `2`
* Estimated Fmax: `655.09 MHz`

## Conclusion

Codex successfully repaired the off-by-one error. This experiment extended the repair workflow from scalar arithmetic to an array-based HLS kernel with an automatically pipelined loop.
