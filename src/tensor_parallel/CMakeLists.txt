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
add_executable(tp_col_nccl.bin
  col_wise.cu
)

target_link_libraries(tp_col_nccl.bin PRIVATE
  CUDA::cudart
  CUDA::cublas
  NCCL::nccl
)

target_compile_options(tp_col_nccl.bin PRIVATE
  ${AICOMM_CUDA_FLAGS}
  ${AICOMM_CXX_FLAGS}
)

# Row-parallel executable
add_executable(tp_row_nccl.bin
  row_wise.cu
)

target_link_libraries(tp_row_nccl.bin PRIVATE
  CUDA::cudart
  CUDA::cublas
  NCCL::nccl
)

target_compile_options(tp_row_nccl.bin PRIVATE
  ${AICOMM_CUDA_FLAGS}
  ${AICOMM_CXX_FLAGS}
)
