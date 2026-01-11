set(CMAKE_CXX_STANDARD 23)
add_library(Aabb aabb.cu aabb.h)

target_link_libraries(Aabb PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(Aabb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Aabb PUBLIC ${PROJECT_SOURCE_DIR}/sources)