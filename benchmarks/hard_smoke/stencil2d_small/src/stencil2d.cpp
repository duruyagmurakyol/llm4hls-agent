#include "stencil2d.h"

void kernel_stencil2d(
    const float input[STENCIL_IN][STENCIL_IN],
    float output[STENCIL_OUT][STENCIL_OUT]) {
#pragma HLS top name = kernel_stencil2d

    for (int i = 0; i < STENCIL_OUT; i++) {
        for (int j = 0; j < STENCIL_OUT; j++) {
            output[i][j] =
                0.50f * input[i + 1][j + 1] +
                0.125f * input[i][j + 1] +
                0.125f * input[i + 2][j + 1] +
                0.125f * input[i + 1][j] +
                0.125f * input[i + 1][j + 2];
        }
    }
}
