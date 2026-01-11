set(CMAKE_CXX_STANDARD 23)
add_library(Bvh bvh.cu bvh.h)

target_link_libraries(Bvh PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(Bvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Bvh PUBLIC ${PROJECT_SOURCE_DIR}/sources)