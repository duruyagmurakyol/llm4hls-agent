# Stream pipeline benchmark

This original benchmark adds a non-matrix, producer-consumer workload to the evaluation set.

The top function processes 64 integers through four concurrent stages connected by bounded `hls::stream` channels:

1. load input values;
2. multiply each value by two;
3. add a constant bias of three;
4. store the result.

The intended behaviour is:

```text
output[i] = 2 * input[i] + 3
```

The supplied source contains one deliberate functional fault in the bias stage: it subtracts three instead of adding three. The repair agent may modify only `src/stream_pipeline.cpp`. The header, testbench and Vitis configuration are protected.

This case exercises behaviour not represented by the matrix-heavy benchmark set:

- multiple helper functions;
- task-level `DATAFLOW`;
- bounded FIFO channels;
- producer-consumer scheduling;
- per-stage pipelining;
- end-to-end functional verification across concurrent stages.
