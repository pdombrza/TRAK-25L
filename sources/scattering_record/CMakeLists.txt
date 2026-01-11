set(CMAKE_CXX_STANDARD 23)
add_library(ScatteringRecord scattering_record.cpp scattering_record.h)

target_link_libraries(ScatteringRecord PRIVATE CUDA::cudart CUDA::cuda_driver glm)

target_include_directories(ScatteringRecord PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ScatteringRecord PUBLIC ${PROJECT_SOURCE_DIR}/sources)