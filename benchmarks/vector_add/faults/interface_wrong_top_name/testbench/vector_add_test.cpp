#include <iostream>
#include "../src/vector_add.h"

int main() {
    const int a[VECTOR_SIZE] = {
        1, -2, 3, 4, 0, 6, -7, 8,
        9, 10, -11, 12, 13, -14, 15, 16
    };
    const int b[VECTOR_SIZE] = {
        16, 15, -14, 13, 12, -11, 10, 9,
        8, -7, 6, 5, -4, 3, 2, 1
    };
    int c[VECTOR_SIZE] = {};

    vector_add(a, b, c);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        const int expected = a[i] + b[i];
        if (c[i] != expected) {
            std::cerr << "FAIL index=" << i
                      << " expected=" << expected
                      << " actual=" << c[i] << '\n';
            return 1;
        }
    }

    std::cout << "Vector-add test passed.\n";
    return 0;
}
