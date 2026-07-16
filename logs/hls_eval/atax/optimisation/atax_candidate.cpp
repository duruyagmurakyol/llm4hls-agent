#include "atax.h"

void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
#pragma HLS top name = kernel_atax
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=1

    const int m = 38;
    const int n = 42;
    const int uf = 14;

    double x_buf[42];
    double y_buf[42];
    double row_buf[42];
    double partial[14];

#pragma HLS ARRAY_PARTITION variable=x_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1

init_loop:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        x_buf[j] = x[j];
        y_buf[j] = 0.0;
    }

row_loop:
    for (int i = 0; i < m; i++) {
#pragma HLS LOOP_FLATTEN off

init_partial:
        for (int k = 0; k < uf; k++) {
#pragma HLS UNROLL
            partial[k] = 0.0;
        }

dot_loop:
        for (int jj = 0; jj < n; jj += uf) {
#pragma HLS PIPELINE II=1
            for (int k = 0; k < uf; k++) {
#pragma HLS UNROLL
                double a_val = A[i][jj + k];
                row_buf[jj + k] = a_val;
                partial[k] += a_val * x_buf[jj + k];
            }
        }

        double tmp_val = 0.0;
reduce_tmp:
        for (int k = 0; k < uf; k++) {
#pragma HLS UNROLL
            tmp_val += partial[k];
        }

        tmp[i] = tmp_val;

update_y:
        for (int jj = 0; jj < n; jj += uf) {
#pragma HLS PIPELINE II=1
            for (int k = 0; k < uf; k++) {
#pragma HLS UNROLL
                y_buf[jj + k] += row_buf[jj + k] * tmp_val;
            }
        }
    }

store_y:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        y[j] = y_buf[j];
    }
}