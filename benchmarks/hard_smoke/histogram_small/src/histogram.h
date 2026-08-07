#pragma once

static const int HIST_N = 64;
static const int HIST_BINS = 16;

void kernel_histogram(
    const unsigned char input[HIST_N],
    unsigned int hist[HIST_BINS]);
