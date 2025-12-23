# FindNCCL.cmake
# Usage:
#   cmake -S . -B build -DNCCL_ROOT=/home/.../nccl/build
#
# Exposes:
#   NCCL::nccl   imported target
#   NCCL_INCLUDE_DIR
#   NCCL_LIBRARY

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS
    ${NCCL_ROOT}
    $ENV{NCCL_ROOT}
  PATH_SUFFIXES include
)

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS
    ${NCCL_ROOT}
    $ENV{NCCL_ROOT}
  PATH_SUFFIXES lib lib64 build/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
)

if(NCCL_FOUND AND NOT TARGET NCCL::nccl)
  add_library(NCCL::nccl UNKNOWN IMPORTED)
  set_target_properties(NCCL::nccl PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
  )
endif()
