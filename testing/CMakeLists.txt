###############################################################################
## symlink for matrices #######################################################
###############################################################################

# Helper function to create relative symbolic links from the current binary directory to the source directory
function(create_relative_symlink_from_bin_dir target link_name)
  # compute relative path from current binary directory to target
  file(RELATIVE_PATH target_rel ${CMAKE_CURRENT_BINARY_DIR} ${target})
  # create symbolic links
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${target_rel} ${CMAKE_CURRENT_BINARY_DIR}/${link_name})
endfunction()

create_relative_symlink_from_bin_dir(${CMAKE_CURRENT_SOURCE_DIR}/matrices matrices)

###############################################################################
## target definitions #########################################################
###############################################################################

add_executable(compare_matrices compare_matrices.cpp ${GLOB_SOURCES})
target_link_libraries(compare_matrices gtest_main)

enable_testing()
add_test(NAME compare_matrices COMMAND compare_matrices)