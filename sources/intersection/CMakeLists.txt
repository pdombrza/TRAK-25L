set(CMAKE_CXX_STANDARD 23)
add_library(Intersection intersection.cu intersection.h)

target_link_libraries(Intersection PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(Intersection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Intersection PUBLIC ${PROJECT_SOURCE_DIR}/sources)