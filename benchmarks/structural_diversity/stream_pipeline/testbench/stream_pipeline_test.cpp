#include "stream_pipeline.h"

#include <iostream>

int main() {
    int input[STREAM_PIPELINE_SIZE];
    int output[STREAM_PIPELINE_SIZE] = {};

    for (int i = 0; i < STREAM_PIPELINE_SIZE; ++i) {
        input[i] = (i % 11) - 5;
    }

    stream_pipeline(input, output);

    for (int i = 0; i < STREAM_PIPELINE_SIZE; ++i) {
        const int expected = input[i] * 2 + 3;
        if (output[i] != expected) {
            std::cerr
                << "Mismatch at index " << i
                << ": expected " << expected
                << ", got " << output[i] << '\n';
            return 1;
        }
    }

    std::cout << "All stream pipeline tests passed.\n";
    return 0;
}
