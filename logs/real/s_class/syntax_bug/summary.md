# Real S-Class Syntax Repair

## Objective

Test whether Codex could repair an existing compile-stage HLS benchmark failure.

## Fault

The kernel was missing a semicolon:

```cpp
out[i] = a[i] + b[i]
```

Vitis HLS reported:

```text
error: expected ';' after expression
```

## Repair

Codex added the missing semicolon while preserving the loop and HLS pipeline pragma.

## Result

* Initial C simulation: Fail
* Repaired C simulation: Pass
* Testbench output: `PASS`
* Repair attempts: 1

## Conclusion

Codex successfully repaired a real S-class syntax benchmark and restored successful Vitis HLS C simulation.
