#ifndef HLS_EVAL_VECTOR_ADD_H
#define HLS_EVAL_VECTOR_ADD_H

constexpr int VECTOR_SIZE = 1024;

void vector_add(
    const float a[VECTOR_SIZE],
    const float b[VECTOR_SIZE],
    float c[VECTOR_SIZE]
);

#endif
