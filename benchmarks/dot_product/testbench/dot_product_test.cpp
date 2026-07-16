#include <iostream>
#include "../src/dot_product.h"

int main() {
    const int a[VECTOR_SIZE] = {
        1, 2, 3, 4, -5, 0, 10, -2
    };

    const int b[VECTOR_SIZE] = {
        8, 7, 6, 5, 2, 9, -3, 6
    };

    const int expected =
        (1 * 8) +
        (2 * 7) +
        (3 * 6) +
        (4 * 5) +
        (-5 * 2) +
        (0 * 9) +
        (10 * -3) +
        (-2 * 6);

    int result = 0;

    dot_product(a, b, result);

    if (result != expected) {
        std::cerr
            << "FAIL: expected " << expected
            << ", got " << result
            << '\n';

        return 1;
    }

    std::cout
        << "Dot product test passed. Result = "
        << result
        << '\n';

    return 0;
}