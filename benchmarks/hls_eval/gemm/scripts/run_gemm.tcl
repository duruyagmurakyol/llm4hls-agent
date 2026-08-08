open_project -reset gemm_hls
set_top kernel_gemm
add_files -cflags "-I/home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/gemm/src" "/home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/gemm/src/gemm.cpp"
add_files "/home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/gemm/src/gemm.h"
add_files -tb -cflags "-I/home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/gemm/src" "/home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/gemm/testbench/gemm_tb.cpp"
open_solution -reset solution1
set_part xcu55c-fsvh2892-2L-e
create_clock -period 10 -name default
