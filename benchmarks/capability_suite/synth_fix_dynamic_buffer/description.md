# Scale-copy synthesis-fix task

`scale_copy` reads `N` signed integers and writes twice each input value to the
corresponding output location:

```text
out[i] = 2 * in[i]
```

The supplied implementation is functionally correct in C-simulation but is not
accepted by HLS synthesis. Preserve the declared interface and numerical
behavior. Use the Vitis synthesis diagnostics to make the implementation
synthesizable without changing the header or testbench.
