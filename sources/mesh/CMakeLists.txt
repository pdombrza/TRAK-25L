set(CMAKE_CXX_STANDARD 23)
add_library(MeshLoader mesh.cpp mesh.h)

target_link_libraries(MeshLoader PRIVATE glm)

target_include_directories(MeshLoader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MeshLoader PUBLIC ${PROJECT_SOURCE_DIR}/sources)
