add_library(Kalman STATIC 
        "kalman.cpp"
        "matrix.hpp"
)

target_include_directories(Kalman 
        PUBLIC 
        "."
)

set(CMAKE_CXX_STANDARD 23)

############### EIGEN LIB ################
target_link_directories(Kalman PRIVATE eigen)
target_link_libraries(Kalman PRIVATE eigen)