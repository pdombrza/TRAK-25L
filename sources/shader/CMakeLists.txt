set(CMAKE_CXX_STANDARD 23)
add_library(Shader shader.cpp shader.h)

target_link_libraries(Shader PRIVATE CUDA::cudart CUDA::cuda_driver glm glfw glad)

target_include_directories(Shader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Shader PUBLIC ${PROJECT_SOURCE_DIR}/sources)