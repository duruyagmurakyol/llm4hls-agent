#include "atax.h"

void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
#pragma HLS top name = kernel_atax

    const int m = 38;
    const int n = 42;

    double x_local[42];
    double y_local[42];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=2 dim=1

    int i, j;

    load_init:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE
        x_local[i] = x[i];
        y_local[i] = 0.0;
    }

    row_loop:
    for (i = 0; i < m; i++) {
        double acc = 0.0;

        dot_loop:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE
            acc = acc + A[i][j] * x_local[j];
        }

        tmp[i] = acc;

        update_y:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=2
            y_local[j] = y_local[j] + A[i][j] * acc;
        }
    }

    store_y:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE
        y[i] = y_local[i];
    }
}
