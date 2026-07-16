#include "atax.h"

void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
#pragma HLS top name = kernel_atax

    const int m = 38;
    const int n = 42;

    double x_buf[42];
    double y_buf[42];
    double a_row[42];

#pragma HLS ARRAY_PARTITION variable=x_buf cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=y_buf cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=a_row cyclic factor=2 dim=1

    int i, j;

    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        x_buf[j] = x[j];
        y_buf[j] = 0.0;
    }

    for (i = 0; i < m; i++) {
        double tmp_val = 0.0;

        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            a_row[j] = A[i][j];
        }

        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            tmp_val = tmp_val + a_row[j] * x_buf[j];
        }

        tmp[i] = tmp_val;

        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
            y_buf[j] = y_buf[j] + a_row[j] * tmp_val;
        }
    }

    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        y[j] = y_buf[j];
    }
}