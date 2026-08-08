open_project -reset vector_add_hls
set_top vector_add
add_files src/vector_add.cpp -cflags "-Isrc"
open_solution -reset solution1 -flow_target vivado
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default
csynth_design
exit
