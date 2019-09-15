cmake_minimum_required(VERSION 3.0.2)
if(POLICY CMP0007)
  cmake_policy(SET CMP0007 NEW)
endif()

project(InstanceFusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Flann REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(PythonLibs REQUIRED)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/elasticfusionpublic/Core/src")

file(GLOB_RECURSE srcs src/*.cpp)
file(GLOB_RECURSE cuda src/*.cu)

set(CUDA_ARCH_BIN "50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")
include(CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
#set(CUDA_HOST_COMPILER "/usr/bin/g++-4.9")
set(CUDA_HOST_COMPILER gcc-5)

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11")

cuda_add_library(Instance_fusion_cuda_library SHARED ${cuda} )

add_executable(${CMAKE_PROJECT_NAME}
               ${srcs}
)

target_link_libraries(${CMAKE_PROJECT_NAME}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
 		      ${PYTHON_LIBRARIES}
                      ${FLANN_LIBRARIES}
		      /usr/local/lib/libflann_cuda.so		#flann_cuda
                      Instance_fusion_cuda_library
                      efusion
                      ${OpenCV_LIBS}
)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                           ${FLANN_INCLUDE_DIRS}
                           ${EIGEN_INCLUDE_DIRS}
                           ${CUDA_INCLUDE_DIRS}
                           ${OPENNI2_INCLUDE_DIR}
		           ${PYTHON_INCLUDE_DIRS}
)


