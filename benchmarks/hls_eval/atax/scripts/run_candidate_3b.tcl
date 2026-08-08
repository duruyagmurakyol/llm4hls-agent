open_project -reset atax_candidate_3b_project
set_top kernel_atax

add_files ../candidates/atax_candidate_3b.cpp
add_files ../candidates/atax.h
add_files -tb -cflags "-I../src" ../testbench/atax_tb.cpp

open_solution -reset solution1

set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default

csim_design
csynth_design

exit
