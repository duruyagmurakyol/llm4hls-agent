#include "atax.h"

void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
#pragma HLS top name = kernel_atax
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=4 dim=1

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
        double acc0 = 0.0;
        double acc1 = 0.0;
        double acc2 = 0.0;
        double acc3 = 0.0;

        dot_loop:
        for (j = 0; j < 40; j += 4) {
#pragma HLS PIPELINE
            acc0 = acc0 + A[i][j]     * x[j];
            acc1 = acc1 + A[i][j + 1] * x[j + 1];
            acc2 = acc2 + A[i][j + 2] * x[j + 2];
            acc3 = acc3 + A[i][j + 3] * x[j + 3];
        }

        acc0 = acc0 + A[i][40] * x[40];
        acc1 = acc1 + A[i][41] * x[41];

        double acc = (acc0 + acc1) + (acc2 + acc3);
        tmp[i] = acc;

        update_y:
        for (j = 0; j < n; j += 4) {
#pragma HLS PIPELINE
            y[j]     = y[j]     + A[i][j]     * acc;
            y[j + 1] = y[j + 1] + A[i][j + 1] * acc;
            y[j + 2] = y[j + 2] + A[i][j + 2] * acc;
            y[j + 3] = y[j + 3] + A[i][j + 3] * acc;
        }
    }
}
