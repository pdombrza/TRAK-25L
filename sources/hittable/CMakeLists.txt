set(CMAKE_CXX_STANDARD 23)
add_library(Hittable hittable.cu hittable.h)

target_link_libraries(Hittable PRIVATE CUDA::cudart CUDA::cuda_driver glm Ray)

target_include_directories(Hittable PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Hittable PUBLIC ${PROJECT_SOURCE_DIR})
