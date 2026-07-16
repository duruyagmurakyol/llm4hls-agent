#include <iostream>
#include "../src/add.h"

int main() {
    int failures = 0;

    struct TestCase {
        int a;
        int b;
        int expected;
    };

    const TestCase tests[] = {
        {4, 7, 11},
        {0, 0, 0},
        {-3, 8, 5},
        {-5, -6, -11},
        {100, 200, 300}
    };

    for (const TestCase &test : tests) {
        int result = 0;
        add(test.a, test.b, result);

        if (result != test.expected) {
            std::cerr
                << "FAIL: "
                << test.a << " + " << test.b
                << " expected " << test.expected
                << ", got " << result
                << '\n';

            failures++;
        }
    }

    if (failures > 0) {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All adder tests passed.\n";
    return 0;
}