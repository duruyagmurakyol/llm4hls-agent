open_project -reset /tmp/llm4hls-agent/vector_add_baseline
set_top vector_add
add_files -cflags "-I/home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/vector_add/src" /home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/vector_add/src/vector_add.cpp
add_files /home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/vector_add/src/vector_add.h
add_files -tb -cflags "-I/home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/vector_add/src" /home/xilinx/projects/llm4hls-agent/benchmarks/hls_eval/vector_add/testbench/vector_add_tb.cpp
open_solution -reset solution1
set_part xcu55c-fsvh2892-2L-e
create_clock -period 10 -name default
csim_design
csynth_design
exit
