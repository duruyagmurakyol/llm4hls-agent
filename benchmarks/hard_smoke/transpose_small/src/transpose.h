#pragma once

static const int TRANSPOSE_N = 16;

void kernel_transpose(
    const float input[TRANSPOSE_N][TRANSPOSE_N],
    float output[TRANSPOSE_N][TRANSPOSE_N]);
