cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED MAKE_PROJECT_NAME)
    set(MAKE_PROJECT_NAME kalman)
endif()

project(${MAKE_PROJECT_NAME})

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 23)

add_library(Lib STATIC
    "lib/kalman.hpp"
    "lib/matrix.hpp"
    "lib/vector3d.hpp"
    "lib/vector6d.hpp"
    "lib/rotation3d.hpp"
    "lib/quaternion3d.hpp"
    "lib/dijkstra.hpp"
    "lib/vector.hpp"
    "lib/lqr.hpp"
)

set_target_properties(Lib PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Lib PUBLIC
    "lib"
)

find_package(fmt REQUIRED)
target_link_libraries(Lib PUBLIC
    fmt::fmt
)

add_executable(App
    "app/main.cpp"
)

target_link_libraries(App PRIVATE
    Lib
)

target_compile_options(Lib PUBLIC
    -std=c++23
    -fconcepts
    -Wall
    -Wextra
    -Wconversion
    -Wshadow
    -Wpedantic
    -Wnarrowing
    -Waddress
    -Wnon-virtual-dtor
    -pedantic
    -Wpessimizing-move
    -Wsign-conversion
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wnull-dereference
    -Wuseless-cast
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wcast-align
)
