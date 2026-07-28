#include "atax.h"

void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
#pragma HLS top name = kernel_atax
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=2 dim=1

    const int m = 38;
    const int n = 42;

    int i, j;

    init_y:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE
        y[i] = 0;
    }

    phase1_tmp:
    for (i = 0; i < m; i++) {
        double acc0 = 0.0;
        double acc1 = 0.0;

        dot_loop:
        for (j = 0; j < 42; j += 2) {
#pragma HLS PIPELINE
            acc0 = acc0 + A[i][j]     * x[j];
            acc1 = acc1 + A[i][j + 1] * x[j + 1];
        }

        tmp[i] = acc0 + acc1;
    }

    phase2_y:
    for (j = 0; j < n; j++) {
        double s0 = 0.0;
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;

        col_loop:
        for (i = 0; i < 38; i += 4) {
#pragma HLS PIPELINE
            s0 = s0 + A[i][j]     * tmp[i];
            s1 = s1 + A[i + 1][j] * tmp[i + 1];
            s2 = s2 + A[i + 2][j] * tmp[i + 2];
            s3 = s3 + A[i + 3][j] * tmp[i + 3];
        }

        y[j] = y[j] + ((s0 + s1) + (s2 + s3));
    }
}
