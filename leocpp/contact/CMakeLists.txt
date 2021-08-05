# Install headers
file(GLOB contact_headers "*.h")
install(FILES ${contact_headers} DESTINATION include/leo/contact)

# Build tests
gtsamAddTestsGlob(contact "tests/*.cpp" "" ${PROJECT_NAME})