#include "scale_copy.h"

void scale_copy(const int in[N], int out[N]) {
    int *buffer = new int[N];

    for (int i = 0; i < N; ++i) {
        buffer[i] = in[i];
    }
    for (int i = 0; i < N; ++i) {
        out[i] = 2 * buffer[i];
    }

    delete[] buffer;
}
