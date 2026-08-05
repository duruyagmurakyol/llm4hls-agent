#include "vector_add.h"

void vector_add(
    const float a[VECTOR_SIZE],
    const float b[VECTOR_SIZE],
    float c[VECTOR_SIZE]
) {
vector_add_loop:
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        c[i] = a[i] + b[i];
    }
}
