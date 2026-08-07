#include "histogram.h"
#include <cstdio>

int main() {
    unsigned char input[HIST_N];
    unsigned int hist[HIST_BINS];
    unsigned int expected[HIST_BINS] = {};

    for (int i = 0; i < HIST_N; i++) {
        input[i] = static_cast<unsigned char>((i * 7 + 3) & 15);
        expected[input[i] & 15]++;
    }

    kernel_histogram(input, hist);

    for (int b = 0; b < HIST_BINS; b++) {
        if (hist[b] != expected[b]) {
            std::printf(
                "Bin %d: expected %u got %u\n",
                b, expected[b], hist[b]);
            return 1;
        }
    }

    std::puts("histogram test passed");
    return 0;
}
