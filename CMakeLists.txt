cmake_minimum_required(VERSION 3.20)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:>")


project(raytracer LANGUAGES CXX CUDA)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

add_executable("${CMAKE_PROJECT_NAME}")

set_property(TARGET "${CMAKE_PROJECT_NAME}" PROPERTY CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# glm
add_subdirectory(thirdparty/glm)
add_subdirectory(thirdparty/glfw)
add_subdirectory(thirdparty/glad)

# CUDA 
find_package(CUDAToolkit REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)	
	enable_language(CUDA)
	find_package(CUDAToolkit)
else()
	message(STATUS "No CUDA compiler found")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(OptiX90 REQUIRED)
message(STATUS "OptiX90 include dir: ${OPTIX90_INCLUDE_DIR}")

file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

file(GLOB_RECURSE MODULE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/sources/*")

# here add subdirectory created in your module
add_subdirectory(sources/ray)
add_subdirectory(sources/utils)
add_subdirectory(sources/kernel)
add_subdirectory(sources/shader)
add_subdirectory(sources/hittable)
add_subdirectory(sources/hittablelist)


target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC OPTIX_USE_STUBS)
add_compile_definitions(SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC SHADERS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/shaders/")

target_sources("${CMAKE_PROJECT_NAME}" PRIVATE ${MY_SOURCES} )

if(MSVC) # Microsoft Visual Studio
	target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

# CUDA
set(GENCODE_FLAGS "-gencode arch=compute_120,code=sm_120")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>>
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo ${GENCODE_FLAGS}>>
)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE OPTIX_USE_STUBS)

target_include_directories("${CMAKE_PROJECT_NAME}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC "${OPTIX90_INCLUDE_DIR}")

target_link_libraries("${CMAKE_PROJECT_NAME}" PRIVATE Kernel Ray Shader Hittable)

target_link_libraries("${CMAKE_PROJECT_NAME}" PRIVATE glm glfw glad CUDA::cudart CUDA::cuda_driver)