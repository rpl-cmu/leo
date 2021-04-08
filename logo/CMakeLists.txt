set(logo_subdirs 
contact
dynamics
geometry
utils
)
set(logo_srcs)

# files want to be excluded
set(excluded_sources "")

# Library sources
foreach(subdir ${logo_subdirs})
  file(GLOB subdir_srcs "${subdir}/*.cpp" "${subdir}/*.h")
  list(REMOVE_ITEM subdir_srcs "${excluded_sources}")

  file(GLOB subdir_test_files "${subdir}/tests/*")
  
  list(APPEND logo_srcs ${subdir_srcs})
  message(STATUS "Building Module: ${subdir}")

  # local and tests
  add_subdirectory(${subdir})
endforeach(subdir)

# build shared lib
message(STATUS "Build shared library")
add_library(${PROJECT_NAME} SHARED ${logo_srcs})
target_link_libraries(logo ${Boost_LIBRARIES} ${GTSAM_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(logo PROPERTIES
  OUTPUT_NAME         ${PROJECT_NAME}
  LINKER_LANGUAGE CXX
  CLEAN_DIRECT_OUTPUT 1)
  install(TARGETS ${PROJECT_NAME} EXPORT logo-exports LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)

list(APPEND LOGO_EXPORTED_TARGETS logo)
set(LOGO_EXPORTED_TARGETS "${LOGO_EXPORTED_TARGETS}" PARENT_SCOPE)