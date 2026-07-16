#include "kernel.h"
void vadd(const int a[N], const int b[N], int out[N]) {
  for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
    out[i] = a[i] + b[i];
  }
}
