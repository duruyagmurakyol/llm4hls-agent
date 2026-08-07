#include "transpose.h"

void kernel_transpose(
    const float input[TRANSPOSE_N][TRANSPOSE_N],
    float output[TRANSPOSE_N][TRANSPOSE_N]) {
#pragma HLS top name = kernel_transpose

    for (int i = 0; i < TRANSPOSE_N; i++) {
        for (int j = 0; j < TRANSPOSE_N; j++) {
            output[j][i] = input[i][j];
        }
    }
}
