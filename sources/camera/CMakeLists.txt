set(CMAKE_CXX_STANDARD 23)
add_library(Camera camera.cu camera.h)

target_link_libraries(Camera PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(Camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Camera PUBLIC ${PROJECT_SOURCE_DIR}/sources)