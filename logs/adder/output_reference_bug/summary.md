# Output Reference Bug Repair Experiment

## Objective

Test whether Codex could identify and repair a C++ interface bug in a Xilinx HLS adder.

## Fault Introduced

The output parameter was changed from a reference:

```cpp
int &result
```

to a value parameter:

```cpp
int result
```

This meant the function updated only a local copy, so the caller continued to see `0`.

## Failure Result

Vitis HLS C simulation failed on four non-zero test cases because every result remained `0`. The `0 + 0` case passed accidentally.

## Repair Result

Codex changed the output parameter back to a reference in both the header and source files.

The repaired version then completed successfully:

* C simulation: Pass
* C synthesis: Pass
* RTL generation: Pass
* Estimated Fmax: `984.25 MHz`

## Configuration

* Tool: Vitis HLS 2025.2
* FPGA part: `xczu3eg-sfvc784-2-e`
* Target clock: `10 ns`
* Top function: `add`

## Conclusion

Codex successfully repaired the pass-by-value bug. This experiment was less obvious than the subtraction bug because the arithmetic was correct, but the function interface prevented the result from reaching the caller.
