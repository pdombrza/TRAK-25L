set(CMAKE_CXX_STANDARD 23)
add_library(Kernel kernel.cu kernel.h)

target_link_libraries(Kernel PRIVATE CUDA::cudart CUDA::cuda_driver glm Ray Hittable HittableList Utils)

target_include_directories(Kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Kernel PUBLIC ${PROJECT_SOURCE_DIR})
