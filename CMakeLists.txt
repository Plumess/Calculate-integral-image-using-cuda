cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(Integral_Img LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB CU_SOURCES ./*.cu)
file(GLOB HEADERS ./*.h ./*.cuh)

add_executable(Integral_Img ${CU_SOURCES} ${CXX_SOURCES} ${HEADERS})
set_target_properties(Integral_Img PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(OpenCV_DIR "E:/OpenCV/install") # 根据你的实际情况修改此处路径
find_package(OpenCV REQUIRED)
target_link_libraries(Integral_Img ${OpenCV_LIBS})

set(CMAKE_BUILD_TYPE Debug CACHE STRING "set build type to debug")

