set(CMAKE_CXX_STANDARD 23)
add_library(Framebuffer framebuffer.cu framebuffer.h)

target_link_libraries(Framebuffer PRIVATE CUDA::cudart CUDA::cuda_driver glm Ray Utils HittableList)

target_include_directories(Framebuffer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Framebuffer PUBLIC ${PROJECT_SOURCE_DIR})
