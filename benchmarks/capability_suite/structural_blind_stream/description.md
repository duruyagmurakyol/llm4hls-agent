# Streaming triple task

`stream_triple` processes `N` input elements in a dataflow implementation. For
every valid index:

```text
out[i] = 3 * in[i]
```

Preserve the top-level interface and exact numerical behavior. The implementation
must pass C-simulation, synthesize, and complete C/RTL co-simulation. Diagnose
any structural problem from the Vitis tool evidence. The public description does
not prescribe a particular internal stream topology or repair.
