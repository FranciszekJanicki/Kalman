add_library(Kalman STATIC
        "kalman.cpp"
        "matrix.hpp"
        "point3d.hpp"
        "quaternion.hpp"
)

target_include_directories(Kalman
        PUBLIC
        "."
)

set(CMAKE_CXX_STANDARD 23)

target_compile_options(Kalman PUBLIC
        -std=c++2b
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
