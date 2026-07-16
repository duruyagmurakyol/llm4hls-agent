#include "vector_add.h"
void vector_add(
    const int a[VECTOR_SIZE],
    const int b[VECTOR_SIZE],
    int result[VECTOR_SIZE]
) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        result[i] = a[i] + b[i];
    }
}