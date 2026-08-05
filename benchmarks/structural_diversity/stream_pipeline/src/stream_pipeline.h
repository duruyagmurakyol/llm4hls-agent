#ifndef STREAM_PIPELINE_H
#define STREAM_PIPELINE_H

constexpr int STREAM_PIPELINE_SIZE = 64;

void stream_pipeline(
    const int input[STREAM_PIPELINE_SIZE],
    int output[STREAM_PIPELINE_SIZE]
);

#endif
