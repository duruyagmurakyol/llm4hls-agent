#include "stream_triple.h"

#include <iostream>

int main() {
    data_t in[N];
    data_t out[N] = {};

    for (int i = 0; i < N; ++i) {
        in[i] = i - 11;
    }

    stream_triple(in, out);

    for (int i = 0; i < N; ++i) {
        const data_t expected = 3 * in[i];
        if (out[i] != expected) {
            std::cout << "Mismatch at " << i << ": expected " << expected
                      << ", got " << out[i] << std::endl;
            return 1;
        }
    }

    std::cout << "All structural tests passed." << std::endl;
    return 0;
}
