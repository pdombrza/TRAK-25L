set(CMAKE_CXX_STANDARD 23)
add_library(Renderer renderer.cu renderer.h)

target_link_libraries(Renderer PRIVATE CUDA::cudart CUDA::cuda_driver glm glfw glad)

add_compile_definitions(SOURCE_ROOT="${CMAKE_SOURCE_DIR}")

target_include_directories(Renderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Renderer PUBLIC ${PROJECT_SOURCE_DIR}/sources)