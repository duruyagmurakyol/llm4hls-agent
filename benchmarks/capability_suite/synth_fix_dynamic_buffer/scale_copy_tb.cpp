#include "scale_copy.h"

#include <iostream>

int main() {
    int in[N];
    int out[N] = {};

    for (int i = 0; i < N; ++i) {
        in[i] = i - 5;
    }

    scale_copy(in, out);

    for (int i = 0; i < N; ++i) {
        const int expected = 2 * in[i];
        if (out[i] != expected) {
            std::cout << "Mismatch at " << i << ": expected " << expected
                      << ", got " << out[i] << std::endl;
            return 1;
        }
    }

    std::cout << "All synthesis-fix tests passed." << std::endl;
    return 0;
}
