#include "prefix_sum.h"
#include <cstdio>

int main() {
    int input[PREFIX_N];
    int output[PREFIX_N];

    for (int i = 0; i < PREFIX_N; i++)
        input[i] = (i % 7) - 3;

    kernel_prefix_sum(input, output);

    int running = 0;
    for (int i = 0; i < PREFIX_N; i++) {
        running += input[i];

        if (output[i] != running) {
            std::printf("Mismatch at %d\n", i);
            return 1;
        }
    }

    std::puts("prefix sum test passed");
    return 0;
}
