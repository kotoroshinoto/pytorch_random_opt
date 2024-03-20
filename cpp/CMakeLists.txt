cmake_minimum_required(VERSION 3.12)
project(pytorch_random_opt_cpp)

# Find Python
find_package(Python REQUIRED COMPONENTS Development)
#include_directories(${Python_INCLUDE_DIRS})

# Find pybind11 headers
find_path(PYBIND11_INCLUDE_DIR NAMES pybind11/pybind11.h PATHS /path/to/anaconda/env/include)
if (NOT PYBIND11_INCLUDE_DIR)
    message(FATAL_ERROR "pybind11 headers not found")
endif()
#include_directories(${PYBIND11_INCLUDE_DIR})

# Add the C++ source files
add_library(my_module SHARED
        src/pybind_module.cpp
)

# Link against Python and pybind11
target_include_directories(my_module PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(my_module PUBLIC ${PYBIND11_INCLUDE_DIR})

# Set C++ standard
set_target_properties(my_module PROPERTIES CXX_STANDARD 20)

# Set output directory
set_target_properties(my_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)

# Optional: Install the shared library
install(TARGETS my_module LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python)
