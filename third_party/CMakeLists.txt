add_subdirectory(eigen)
add_library(eigen3 STATIC IMPORTED GLOBAL)
set_target_properties(eigen3 PROPERTIES
    IMPORTED_LOCATION "${THIRD_PARTY_DIR}/eigen"
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

include_directories(${EIGEN3_INCLUDE_DIR})