<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" top="vector_add" name="vector_add_hls" ideType="classic">
    <files>
        <file name="src/vector_add.cpp" sc="0" tb="false" cflags="-Isrc" csimflags="" blackbox="false"/>
        <file name="../../testbench/vector_add_test.cpp" sc="0" tb="1" cflags="-I../../src -Wno-unknown-pragmas" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>
