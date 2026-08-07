#include "histogram.h"

void kernel_histogram(
    const unsigned char input[HIST_N],
    unsigned int hist[HIST_BINS]) {
#pragma HLS top name = kernel_histogram

    for (int b = 0; b < HIST_BINS; b++) {
        hist[b] = 0;
    }

    for (int i = 0; i < HIST_N; i++) {
        unsigned int bin = input[i] & 15;
        hist[bin]++;
    }
}
