#include "atax.h"

void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
#pragma HLS top name = kernel_atax
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=2 dim=2

    const int m = 38;
    const int n = 42;

    int i, j;

    init_y:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE
        y[i] = 0;
    }

    row_loop:
    for (i = 0; i < m; i++) {
        double acc = 0.0;

        dot_loop:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE
            acc = acc + A[i][j] * x[j];
        }

        tmp[i] = acc;

        update_y:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=2
            y[j] = y[j] + A[i][j] * acc;
        }
    }
}
