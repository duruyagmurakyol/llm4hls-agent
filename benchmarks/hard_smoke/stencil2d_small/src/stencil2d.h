#pragma once

static const int STENCIL_IN = 16;
static const int STENCIL_OUT = 14;

void kernel_stencil2d(
    const float input[STENCIL_IN][STENCIL_IN],
    float output[STENCIL_OUT][STENCIL_OUT]);
