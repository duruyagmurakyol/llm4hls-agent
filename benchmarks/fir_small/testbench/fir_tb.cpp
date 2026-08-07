#include "fir.h"

#include <cmath>
#include <cstdio>

int main() {
    float x[64];
    float h[8];
    float y[64];
    float expected[64];

    for (int i = 0; i < 64; i++) {
        x[i] = static_cast<float>((i % 11) - 5) * 0.25f;
        y[i] = 0.0f;
        expected[i] = 0.0f;
    }

    for (int k = 0; k < 8; k++) {
        h[k] = static_cast<float>(k + 1) * 0.05f;
    }

    for (int n = 0; n < 64; n++) {
        float acc = 0.0f;

        for (int k = 0; k < 8; k++) {
            if (n >= k) {
                acc += x[n - k] * h[k];
            }
        }

        expected[n] = acc;
    }

    kernel_fir(x, h, y);

    for (int i = 0; i < 64; i++) {
        if (std::fabs(y[i] - expected[i]) > 1e-5f) {
            std::printf(
                "Mismatch at %d: got %.8f expected %.8f\n",
                i,
                y[i],
                expected[i]);
            return 1;
        }
    }

    std::printf("All FIR tests passed.\n");
    return 0;
}
