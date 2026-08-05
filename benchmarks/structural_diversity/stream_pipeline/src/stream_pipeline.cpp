#include "stream_pipeline.h"

#include <hls_stream.h>

namespace {

void load_input(
    const int input[STREAM_PIPELINE_SIZE],
    hls::stream<int> &loaded
) {
    for (int i = 0; i < STREAM_PIPELINE_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        loaded.write(input[i]);
    }
}

void scale_values(
    hls::stream<int> &loaded,
    hls::stream<int> &scaled
) {
    for (int i = 0; i < STREAM_PIPELINE_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        scaled.write(loaded.read() * 2);
    }
}

void add_bias(
    hls::stream<int> &scaled,
    hls::stream<int> &biased
) {
    for (int i = 0; i < STREAM_PIPELINE_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        // Deliberate functional fault: the required bias is +3.
        biased.write(scaled.read() - 3);
    }
}

void store_output(
    hls::stream<int> &biased,
    int output[STREAM_PIPELINE_SIZE]
) {
    for (int i = 0; i < STREAM_PIPELINE_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = biased.read();
    }
}

}  // namespace

void stream_pipeline(
    const int input[STREAM_PIPELINE_SIZE],
    int output[STREAM_PIPELINE_SIZE]
) {
#pragma HLS DATAFLOW

    hls::stream<int> loaded("loaded");
    hls::stream<int> scaled("scaled");
    hls::stream<int> biased("biased");
#pragma HLS STREAM variable=loaded depth=4
#pragma HLS STREAM variable=scaled depth=4
#pragma HLS STREAM variable=biased depth=4

    load_input(input, loaded);
    scale_values(loaded, scaled);
    add_bias(scaled, biased);
    store_output(biased, output);
}
