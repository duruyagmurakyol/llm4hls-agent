# Dot Product Accumulator Bug Repair

## Objective

Test whether Codex could repair an accumulation error in a Xilinx HLS dot-product kernel.

## Fault Introduced

The accumulator update:

```cpp
sum += a[i] * b[i];
```

was changed to:

```cpp
sum = a[i] * b[i];
```

This overwrote the running total on every iteration, leaving only the final multiplication result.

## Failure Result

C simulation failed because the implementation returned `-12` instead of the expected full dot product, `8`.

## Repair Result

Codex restored the accumulation operation.

The repaired implementation passed:

* C simulation
* C synthesis
* RTL generation

## Conclusion

Codex successfully repaired a loop-carried accumulation bug. This extended the workflow to a kernel involving multiplication, reduction and DSP-related synthesis behaviour.
