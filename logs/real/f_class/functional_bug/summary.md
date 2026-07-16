# Real F-Class Functional Repair

## Objective

Test whether Codex could repair an existing functional HLS benchmark failure.

## Fault

The kernel subtracted corresponding vector elements:

```cpp
out[i] = a[i] - b[i];
```

instead of adding them.

The original C simulation completed compilation but failed with:

```text
FAIL 63
```

## Repair

Codex replaced subtraction with addition while preserving the loop and HLS pipeline pragma.

## Result

* Initial C simulation: Fail
* Repaired C simulation: Pass
* Testbench output: `PASS`
* Repair attempts: 1

## Conclusion

Codex successfully repaired a real F-class functional benchmark and restored correct Vitis HLS C simulation.
