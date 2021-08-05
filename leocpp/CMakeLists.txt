set(leocpp_subdirs 
contact
dynamics
geometry
utils
)
set(leocpp_srcs)

# files want to be excluded
set(excluded_sources "")

# Library sources
foreach(subdir ${leocpp_subdirs})
  file(GLOB subdir_srcs "${subdir}/*.cpp" "${subdir}/*.h")
  list(REMOVE_ITEM subdir_srcs "${excluded_sources}")

  file(GLOB subdir_test_files "${subdir}/tests/*")
  
  list(APPEND leocpp_srcs ${subdir_srcs})
  message(STATUS "Building Module: ${subdir}")

  # local and tests
  add_subdirectory(${subdir})
endforeach(subdir)

# shared
message(STATUS "Build shared library")
add_library(${PROJECT_NAME} SHARED ${leocpp_srcs})
target_link_libraries(leocpp ${Boost_LIBRARIES} ${GTSAM_LIBRARIES})
set_target_properties(leocpp PROPERTIES
  OUTPUT_NAME         ${PROJECT_NAME}
  LINKER_LANGUAGE CXX
  CLEAN_DIRECT_OUTPUT 1)
  install(TARGETS ${PROJECT_NAME} EXPORT leocpp-exports LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)

list(APPEND LEOCPP_EXPORTED_TARGETS leocpp)
set(LEOCPP_EXPORTED_TARGETS "${LEOCPP_EXPORTED_TARGETS}" PARENT_SCOPE)