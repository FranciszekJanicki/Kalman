add_library(Kalman STATIC 
        "kalman.cpp"
        "matrix.hpp"
)

target_include_directories(Kalman 
        PUBLIC 
        "."
)

set(CMAKE_CXX_STANDARD 23)

target_compile_options(Kalman PRIVATE
        -std=c++23
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


############### EIGEN LIB ################
target_link_directories(Kalman PRIVATE eigen)
target_link_libraries(Kalman PRIVATE eigen)
