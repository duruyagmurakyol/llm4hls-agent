#include "conv2d.h"

void kernel_conv2d(
    const float input[CONV_IN][CONV_IN],
    const float kernel[CONV_K][CONV_K],
    float output[CONV_OUT][CONV_OUT]) {
#pragma HLS top name = kernel_conv2d

    for (int i = 0; i < CONV_OUT; i++) {
        for (int j = 0; j < CONV_OUT; j++) {
            float acc = 0.0f;
            for (int ki = 0; ki < CONV_K; ki++) {
                for (int kj = 0; kj < CONV_K; kj++) {
                    acc += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = acc;
        }
    }
}
