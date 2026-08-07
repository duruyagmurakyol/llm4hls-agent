#pragma once

static const int PREFIX_N = 64;

void kernel_prefix_sum(
    const int input[PREFIX_N],
    int output[PREFIX_N]);
