set(CMAKE_CXX_STANDARD 23)
add_library(Utils "utils.cu" utils.h)

target_link_libraries(Utils PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(Utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Utils PUBLIC ${PROJECT_SOURCE_DIR}/sources)