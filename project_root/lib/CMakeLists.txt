add_library(Kalman STATIC 
        "src/kalman.cpp"
        "src/matrix.hpp"
)

target_include_directories(Kalman 
        PUBLIC "include"
        PRIVATE "src"
)

set(CMAKE_CXX_STANDARD 23)

############### EIGEN LIB ################
target_link_directories(Kalman PRIVATE eigen)
target_link_libraries(Kalman PRIVATE eigen)