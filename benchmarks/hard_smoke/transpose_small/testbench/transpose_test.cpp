#include "transpose.h"
#include <cstdio>

int main() {
    float input[TRANSPOSE_N][TRANSPOSE_N];
    float output[TRANSPOSE_N][TRANSPOSE_N];

    for (int i = 0; i < TRANSPOSE_N; i++)
        for (int j = 0; j < TRANSPOSE_N; j++)
            input[i][j] = float(i * 100 + j);

    kernel_transpose(input, output);

    for (int i = 0; i < TRANSPOSE_N; i++) {
        for (int j = 0; j < TRANSPOSE_N; j++) {
            if (output[j][i] != input[i][j]) {
                std::printf("Mismatch at %d,%d\n", i, j);
                return 1;
            }
        }
    }

    std::puts("transpose test passed");
    return 0;
}
