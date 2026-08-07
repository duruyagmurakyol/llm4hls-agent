#pragma once

void kernel_fir(
    const float x[64],
    const float h[8],
    float y[64]);
