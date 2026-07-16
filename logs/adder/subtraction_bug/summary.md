# Subtraction Bug Repair Experiment

## Objective

Test whether Codex could repair a simple functional error in a Xilinx HLS adder.

## Fault Introduced

The correct statement:

```cpp
result = a + b;
```

was changed to:

```cpp
result = a - b;
```

## Failure Result

Vitis HLS C simulation failed on four test cases:

```text
FAIL: 4 + 7 expected 11, got -3
FAIL: -3 + 8 expected 5, got -11
FAIL: -5 + -6 expected -11, got 1
FAIL: 100 + 200 expected 300, got -100
```

The `0 + 0` case passed accidentally.

## Prompt Strategy

Codex was given the faulty source code and told that the function should add two integers while preserving the existing interface.

## Repair Result

Codex replaced subtraction with addition in one attempt.

The repaired implementation passed:

* C simulation
* C synthesis
* RTL generation

## Configuration

* Tool: Vitis HLS 2025.2
* FPGA part: `xczu3eg-sfvc784-2-e`
* Target clock: `10 ns`
* Estimated Fmax: approximately `984.25 MHz`

## Conclusion

The experiment successfully validated the basic workflow:

```text
Introduce bug → run C simulation → prompt Codex → repair code → rerun validation
```

The next experiment will use a less obvious pass-by-value interface bug.

