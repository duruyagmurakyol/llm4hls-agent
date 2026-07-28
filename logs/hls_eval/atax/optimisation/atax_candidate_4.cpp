#include "atax.h"

void kernel_atax(
    double A[38][42],
    double x[42],
    double y[42],
    double tmp[38]
) {
#pragma HLS top name=kernel_atax
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=2 dim=2
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
        double acc2 = 0.0;
        double acc3 = 0.0;

dot_loop:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            double product = A[i][j] * x[j];

            switch (j & 3) {
                case 0:
                    acc0 += product;
                    break;
                case 1:
                    acc1 += product;
                    break;
                case 2:
                    acc2 += product;
                    break;
                default:
                    acc3 += product;
                    break;
            }
        }

        double sum01 = acc0 + acc1;
        double sum23 = acc2 + acc3;
        double acc = sum01 + sum23;

        tmp[i] = acc;

update_y:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
            y[j] += A[i][j] * acc;
        }
    }
}