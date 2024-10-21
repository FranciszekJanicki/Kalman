add_library(Kalman STATIC
        "lib/kalman.cpp"
        "lib/matrix.hpp"
        "lib/vector3d.hpp"
        "lib/vector6d.hpp"
        "lib/rotation3d.hpp"
        "lib/quaternion3d.hpp"
        "lib/dijkstra.hpp"
)
target_include_directories(Kalman PUBLIC
        "lib"
)
find_package(fmt REQUIRED)
target_link_libraries(Kalman PUBLIC
        fmt::fmt
)

add_executable(App
        "app/main.cpp")
target_link_libraries(App PRIVATE
        Kalman
)

target_compile_options(Kalman PUBLIC
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
