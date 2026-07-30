#include <iostream>
#include "vector_add.h"

int main() {
    int a[VECTOR_SIZE];
    int b[VECTOR_SIZE];
    int c[VECTOR_SIZE] = {};

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        a[i] = (i * 7) - 13;
        b[i] = (i * 3) + 5;
    }

    vector_add(a, b, c);

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        const int expected = a[i] + b[i];
        if (c[i] != expected) {
            std::cerr << "FAIL index=" << i << " expected=" << expected
                      << " actual=" << c[i] << '\n';
            return 1;
        }
    }

    std::cout << "All vector-add tests passed.\n";
    return 0;
}
