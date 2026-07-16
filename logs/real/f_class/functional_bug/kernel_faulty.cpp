#include "kernel.h"
// F-class fixture: compiles fine but implements the wrong operation (subtraction instead of
// addition) -> passes stage 1 (compile), fails stage 2 (csim self-check).
void vadd(const int a[N], const int b[N], int out[N]) {
  for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
    out[i] = a[i] - b[i];
  }
}
