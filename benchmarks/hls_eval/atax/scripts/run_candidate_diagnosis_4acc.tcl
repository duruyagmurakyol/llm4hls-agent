open_project -reset atax_candidate_diagnosis_4acc_project
set_top kernel_atax

add_files -cflags "-I../src" ../../../../logs/hls_eval/atax/optimisation/atax_candidate_diagnosis_4acc.cpp
add_files ../src/atax.h
add_files -tb -cflags "-I../src" ../testbench/atax_tb.cpp

open_solution -reset solution1

set_part {xczu3eg-sfvc784-2-e}
create_clock -period 10 -name default

csim_design
csynth_design

exit
