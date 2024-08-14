cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED MAKE_PROJECT_NAME)
    set(MAKE_PROJECT_NAME kalman)
endif()

project(${MAKE_PROJECT_NAME})

add_subdirectory(app)
add_subdirectory(lib)
add_subdirectory(third_party)
