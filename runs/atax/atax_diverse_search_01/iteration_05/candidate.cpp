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
#pragma HLS PIPELINE II=1
        y[i] = 0.0;
    }

    row_loop:
    for (i = 0; i < m; i++) {
        double acc0 = 0.0;
        double acc1 = 0.0;

        double x_buf[42];
#pragma HLS ARRAY_PARTITION variable=x_buf cyclic factor=2 dim=1
#pragma HLS STREAM variable=x_buf depth=2 off

        load_x:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            x_buf[j] = x[j];
        }

        dot_loop:
        for (j = 0; j < n; j += 2) {
#pragma HLS PIPELINE II=1
            double a0 = A[i][j];
            double a1 = A[i][j + 1];
            double x0 = x_buf[j];
            double x1 = x_buf[j + 1];
            acc0 = acc0 + a0 * x0;
            acc1 = acc1 + a1 * x1;
        }

        double acc = acc0 + acc1;
        tmp[i] = acc;

        update_y:
        for (j = 0; j < n; j += 2) {
#pragma HLS PIPELINE II=1
            double a0 = A[i][j];
            double a1 = A[i][j + 1];
            double y0 = y[j];
            double y1 = y[j + 1];
            y[j]     = y0 + a0 * acc;
            y[j + 1] = y1 + a1 * acc;
        }
    }
}
