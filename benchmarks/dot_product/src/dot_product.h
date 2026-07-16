#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

constexpr int VECTOR_SIZE = 8;

void dot_product(
    const int a[VECTOR_SIZE],
    const int b[VECTOR_SIZE],
    int &result
);

#endif