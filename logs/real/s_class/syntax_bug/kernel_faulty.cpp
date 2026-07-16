#include "kernel.h"
// S-class fixture: deliberate syntax error (missing semicolon) -> stage 1 (compile) failure.
void vadd(const int a[N], const int b[N], int out[N]) {
  for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
    out[i] = a[i] + b[i]
  }
}
