# Install headers
file(GLOB geometry_headers "*.h")
install(FILES ${geometry_headers} DESTINATION include/leo/geometry)

# Build tests
gtsamAddTestsGlob(geometry "tests/*.cpp" "" ${PROJECT_NAME})