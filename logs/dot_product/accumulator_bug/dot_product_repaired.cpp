#include "dot_product.h"

void dot_product(
    const int a[VECTOR_SIZE],
    const int b[VECTOR_SIZE],
    int &result
) {
    int sum = 0;

    for (int i = 0; i < VECTOR_SIZE; i++) {
        sum += a[i] * b[i];
    }

    result = sum;
}