#include "prefix_sum.h"

void kernel_prefix_sum(
    const int input[PREFIX_N],
    int output[PREFIX_N]) {
#pragma HLS top name = kernel_prefix_sum

    int running = 0;

    for (int i = 0; i < PREFIX_N; i++) {
        running += input[i];
        output[i] = running;
    }
}
