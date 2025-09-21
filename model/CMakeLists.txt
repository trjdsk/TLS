cmake_minimum_required(VERSION 3.13.1)
project(edge_impulse_model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/edge-impulse-sdk
    ${CMAKE_SOURCE_DIR}/model-parameters
    ${CMAKE_SOURCE_DIR}/tflite-model
)

# Collect SDK + model sources
file(GLOB_RECURSE EI_SOURCES
    ${CMAKE_SOURCE_DIR}/edge-impulse-sdk/**/*.cpp
    ${CMAKE_SOURCE_DIR}/tflite-model/*.cpp
)

# Build shared library
add_library(edge_impulse_model SHARED
    model_wrapper.cpp
    ${EI_SOURCES}
)

# Output
set_target_properties(edge_impulse_model PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(UNIX)
    set_target_properties(edge_impulse_model PROPERTIES SUFFIX ".so")
elseif(WIN32)
    set_target_properties(edge_impulse_model PROPERTIES SUFFIX ".dll")
endif()
