set(CMAKE_CXX_STANDARD 23)
add_library(Material material.cu material.h)

target_link_libraries(Material PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(Material PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Material PUBLIC ${PROJECT_SOURCE_DIR}/sources)
