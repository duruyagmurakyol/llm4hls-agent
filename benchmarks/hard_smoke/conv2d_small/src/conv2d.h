#pragma once

static const int CONV_IN = 16;
static const int CONV_K = 3;
static const int CONV_OUT = 14;

void kernel_conv2d(
    const float input[CONV_IN][CONV_IN],
    const float kernel[CONV_K][CONV_K],
    float output[CONV_OUT][CONV_OUT]);
