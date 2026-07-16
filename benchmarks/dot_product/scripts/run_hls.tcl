open_project -reset dot_product_project
set_top dot_product

add_files ../src/dot_product.cpp
add_files -tb ../testbench/dot_product_test.cpp

open_solution -reset solution1

set_part {xczu3eg-sfvc784-2-e}
create_clock -period 10 -name default

csim_design
csynth_design

exit