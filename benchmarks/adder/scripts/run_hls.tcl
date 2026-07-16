open_project -reset adder_project
set_top add

add_files ../src/add.cpp
add_files -tb ../testbench/add_test.cpp

open_solution -reset solution1

set_part {xczu3eg-sfvc784-2-e}
create_clock -period 10 -name default

csim_design
csynth_design

exit