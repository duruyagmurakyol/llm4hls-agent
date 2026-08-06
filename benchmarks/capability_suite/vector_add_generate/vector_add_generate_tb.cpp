#include "vector_add_generate.h"

#include <iostream>

int main() {
    int a[N];
    int b[N];
    int out[N] = {};

    for (int i = 0; i < N; ++i) {
        a[i] = i - 7;
        b[i] = 3 * i + 2;
    }

    vector_add_generate(a, b, out);

    for (int i = 0; i < N; ++i) {
        const int expected = a[i] + b[i];
        if (out[i] != expected) {
            std::cout << "Mismatch at " << i << ": expected " << expected
                      << ", got " << out[i] << std::endl;
            return 1;
        }
    }

    std::cout << "All generation tests passed." << std::endl;
    return 0;
}
