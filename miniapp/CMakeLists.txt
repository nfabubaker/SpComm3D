###############################################################################
## target definitions #########################################################
###############################################################################

add_executable(benchSddmm bench_sddmm.cpp ${GLOB_SOURCES})

add_executable(benchSpMM bench_spmm.cpp ${GLOB_SOURCES})

# add_executable(benchBoth main_combined.cpp ${GLOB_SOURCES})

# add_executable(serialSim serialSim.cpp ${GLOB_SOURCES})

add_executable(numCheckSeq numCheckSeq.cpp ${GLOB_SOURCES})

add_executable(test test.cpp ${GLOB_SOURCES})
