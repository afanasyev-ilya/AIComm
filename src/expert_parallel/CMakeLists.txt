# Common compile flags
set(AICOMM_CUDA_FLAGS "")
set(AICOMM_CXX_FLAGS "")

# You can tune these; keeping conservative:
list(APPEND AICOMM_CUDA_FLAGS
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
list(APPEND AICOMM_CXX_FLAGS
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Column-parallel executable
add_executable(ep_moe.bin
  moe.cu
)

target_link_libraries(ep_moe.bin PRIVATE
  CUDA::cudart
  CUDA::cublas
  NCCL::nccl
)

target_compile_options(ep_moe.bin PRIVATE
  ${AICOMM_CUDA_FLAGS}
  ${AICOMM_CXX_FLAGS}
)

