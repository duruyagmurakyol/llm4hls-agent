#include "fir.h"

void kernel_fir(
    const float x[64],
    const float h[8],
    float y[64]) {
#pragma HLS top name = kernel_fir

    const int N = 64;
    const int TAPS = 8;

    for (int n = 0; n < N; n++) {
        float acc = 0.0f;

        for (int k = 0; k < TAPS; k++) {
            if (n >= k) {
                acc += x[n - k] * h[k];
            }
        }

        y[n] = acc;
    }
}
