open_project -reset vector_add_project
set_top vector_add

add_files ../src/vector_add.cpp
add_files -tb ../testbench/vector_add_test.cpp

open_solution -reset solution1

set_part {xczu3eg-sfvc784-2-e}
create_clock -period 10 -name default

csim_design
csynth_design

exit