#include "stream_triple.h"
#include "hls_stream.h"

static void split_input(
    const data_t in[N],
    hls::stream<data_t> &main_path,
    hls::stream<data_t> &skip_path) {
    for (int i = 0; i < N; ++i) {
        main_path.write(in[i]);
    }
    for (int i = 0; i < N; ++i) {
        skip_path.write(in[i]);
    }
}

static void double_values(
    hls::stream<data_t> &main_path,
    hls::stream<data_t> &scaled_path) {
    for (int i = 0; i < N; ++i) {
        scaled_path.write(2 * main_path.read());
    }
}

static void add_skip(
    hls::stream<data_t> &scaled_path,
    hls::stream<data_t> &skip_path,
    data_t out[N]) {
    for (int i = 0; i < N; ++i) {
        out[i] = scaled_path.read() + skip_path.read();
    }
}

void stream_triple(const data_t in[N], data_t out[N]) {
#pragma HLS DATAFLOW
    hls::stream<data_t> main_path;
    hls::stream<data_t> scaled_path;
    hls::stream<data_t> skip_path;
#pragma HLS STREAM variable=main_path depth=2
#pragma HLS STREAM variable=scaled_path depth=2
#pragma HLS STREAM variable=skip_path depth=2

    split_input(in, main_path, skip_path);
    double_values(main_path, scaled_path);
    add_skip(scaled_path, skip_path, out);
}
