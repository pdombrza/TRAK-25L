set(CMAKE_CXX_STANDARD 23)
add_library(HittableList hittablelist.cu hittablelist.h)

target_link_libraries(HittableList PRIVATE CUDA::cudart CUDA::cuda_driver glm Ray)

target_include_directories(HittableList PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(HittableList PUBLIC ${PROJECT_SOURCE_DIR})
