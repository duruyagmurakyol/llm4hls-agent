#include "conv2d.h"
#include <cmath>
#include <cstdio>

int main() {
    float input[CONV_IN][CONV_IN];
    float kernel[CONV_K][CONV_K];
    float output[CONV_OUT][CONV_OUT];

    for (int i = 0; i < CONV_IN; i++)
        for (int j = 0; j < CONV_IN; j++)
            input[i][j] = float((i * 7 + j * 3) % 19) / 7.0f;

    for (int i = 0; i < CONV_K; i++)
        for (int j = 0; j < CONV_K; j++)
            kernel[i][j] = float(i * CONV_K + j + 1) / 9.0f;

    kernel_conv2d(input, kernel, output);

    for (int i = 0; i < CONV_OUT; i++) {
        for (int j = 0; j < CONV_OUT; j++) {
            float expected = 0.0f;
            for (int ki = 0; ki < CONV_K; ki++)
                for (int kj = 0; kj < CONV_K; kj++)
                    expected += input[i + ki][j + kj] * kernel[ki][kj];

            if (std::fabs(output[i][j] - expected) > 1e-4f) {
                std::printf("Mismatch at %d,%d\n", i, j);
                return 1;
            }
        }
    }

    std::puts("conv2d test passed");
    return 0;
}
