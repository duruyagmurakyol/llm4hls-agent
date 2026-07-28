#include "vector_add.h"

void vector_add(
    const int a[VECTOR_SIZE],
    const int b[VECTOR_SIZE],
    int c[VECTOR_SIZE]
) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        c[i] = a[i] - b[i];
    }
}
