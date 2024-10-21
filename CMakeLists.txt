cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED MAKE_PROJECT_NAME)
    set(MAKE_PROJECT_NAME kalman)
endif()

project(${MAKE_PROJECT_NAME})

add_subdirectory(app)
add_subdirectory(lib)

set(CMAKE_CXX_COMPILER "usr/bin/g++")
set(CMAKE_CXX_STANDARD 23)
