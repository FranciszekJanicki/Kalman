add_library(Kalman STATIC 
        "src/kalman.cpp"
        "src/matrix.hpp"
)

target_include_directories(Kalman 
        PUBLIC "include"
        PRIVATE "src"
)

############### EIGEN LIB ################
target_link_directories(Kalman PRIVATE eigen)
target_link_libraries(Kalman PRIVATE eigen)