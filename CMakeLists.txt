cmake_minimum_required(VERSION 3.0)
enable_testing()
project(logo CXX C)

# specify C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fopenmp")

# options
set (CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Install prefix for library")
option(LOGO_BUILD_PYTHON_TOOLBOX "build python toolbox" ON)

# GTSAM
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)

find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
include(GtsamTesting)

# for unittest scripts
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

# Boost
find_package(Boost 1.50 REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
find_package(Boost COMPONENTS serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# include current source folder and generated config header files
add_definitions(-DBASE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

# Process subdirs
add_subdirectory(logo)

# Python wrapper
if(LOGO_BUILD_PYTHON_TOOLBOX)
  include_directories(${GTSAM_DIR}/cython)
  include_directories(/usr/local/cython)
  include(GtsamCythonWrap)
  include_directories(${GTSAM_EIGENCY_INSTALL_PATH})

  wrap_and_install_library_cython("logo.h"
                                  "from gtsam.gtsam cimport *" # extra import of gtsam/gtsam.pxd Cython header
                                  "${CMAKE_INSTALL_PREFIX}/cython" # install path
                                  logo  # library to link with
                                  "gtsam"  # dependencies which need to be built before wrapping
                                  )
  add_definitions(-DBOOST_OPTIONAL_ALLOW_BINDING_TO_RVALUES -DBOOST_OPTIONAL_CONFIG_ALLOW_BINDING_TO_RVALUES)
endif()

# Install config and export files
GtsamMakeConfigFile(logo)
export(TARGETS ${LOGO_EXPORTED_TARGETS} FILE logo-exports.cmake)