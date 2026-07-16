#include <iostream>
#include "../src/vector_add.h"

int main() {
    const int a[VECTOR_SIZE] = {
        1, 2, 3, 4, -5, 0, 100, -20
    };

    const int b[VECTOR_SIZE] = {
        8, 7, 6, 5, 10, 0, -50, 30
    };

    const int expected[VECTOR_SIZE] = {
        9, 9, 9, 9, 5, 0, 50, 10
    };

    int result[VECTOR_SIZE] = {0};
    int failures = 0;

    vector_add(a, b, result);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (result[i] != expected[i]) {
            std::cerr
                << "FAIL at index " << i
                << ": expected " << expected[i]
                << ", got " << result[i]
                << '\n';

            failures++;
        }
    }

    if (failures > 0) {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All vector addition tests passed.\n";
    return 0;
}