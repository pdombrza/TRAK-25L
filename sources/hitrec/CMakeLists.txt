set(CMAKE_CXX_STANDARD 23)
add_library(HitRecord "hitrec.cu" hitrec.h)

target_link_libraries(HitRecord PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(HitRecord PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(HitRecord PUBLIC ${PROJECT_SOURCE_DIR}/sources)