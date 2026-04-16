# Core library — everything C++/CUDA will live here, compiled into libctd.
# Kept as a single target for now; we can split later if needed.

add_library(ctd STATIC
  core/allocator.cpp
  core/autograd.cu
  core/cublas_utils.cpp
  core/optim.cu
  core/safetensors.cpp
  core/sampling.cpp
  core/storage.cpp
  core/tensor.cpp
  ops/elementwise.cu
  ops/embedding.cu
  ops/kv_cache.cu
  ops/loss.cu
  ops/mask.cu
  ops/matmul.cpp
  ops/norm.cu
  ops/permute.cu
  ops/repeat_kv.cu
  ops/rope.cu
  ops/softmax.cu
  nn/model.cpp
)

target_include_directories(ctd
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ctd
  PUBLIC
    CUDA::cudart
    CUDA::cublas
  PRIVATE
    ctd_warnings
    nlohmann_json::nlohmann_json
)

set_target_properties(ctd PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)
