#include "stencil2d.h"
#include <cmath>
#include <cstdio>

int main() {
    float input[STENCIL_IN][STENCIL_IN];
    float output[STENCIL_OUT][STENCIL_OUT];

    for (int i = 0; i < STENCIL_IN; i++)
        for (int j = 0; j < STENCIL_IN; j++)
            input[i][j] = float((i * 5 + j * 11) % 23);

    kernel_stencil2d(input, output);

    for (int i = 0; i < STENCIL_OUT; i++) {
        for (int j = 0; j < STENCIL_OUT; j++) {
            float expected =
                0.50f * input[i + 1][j + 1] +
                0.125f * input[i][j + 1] +
                0.125f * input[i + 2][j + 1] +
                0.125f * input[i + 1][j] +
                0.125f * input[i + 1][j + 2];

            if (std::fabs(output[i][j] - expected) > 1e-5f) {
                std::printf("Mismatch at %d,%d\n", i, j);
                return 1;
            }
        }
    }

    std::puts("stencil test passed");
    return 0;
}
