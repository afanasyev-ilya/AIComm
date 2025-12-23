cmake_minimum_required(VERSION 3.22)

project(AIComm LANGUAGES CXX CUDA)

# ---- Output: ./build/bin ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# For multi-config generators (Ninja Multi-Config, VS)
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# ---- Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional: better default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optional: set GPU arch; you can override with -DCMAKE_CUDA_ARCHITECTURES=86 etc
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # "native" is supported on newer CMake; if your CMake is older, set an explicit arch.
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA arch" FORCE)
endif()

# ---- CUDA toolkit (cudart, cublas) ----
find_package(CUDAToolkit REQUIRED)

# ---- NCCL ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NCCL REQUIRED)

# ---- Subprojects ----
add_subdirectory(src)
