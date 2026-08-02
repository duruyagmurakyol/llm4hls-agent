#include "vector_add.h"
#include <cmath>
#include <iostream>

int main() {
    float a[VECTOR_SIZE];
    float b[VECTOR_SIZE];
    float c[VECTOR_SIZE] = {};
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        a[i] = static_cast<float>((i % 29) - 14) * 0.25f;
        b[i] = static_cast<float>((i % 17) - 8) * 0.5f;
    }
    vector_add(a, b, c);
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        const float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1.0e-5f) {
            std::cerr << "Mismatch at index " << i << '\n';
            return 1;
        }
    }
    std::cout << "All vector-add tests passed.\n";
    return 0;
}
