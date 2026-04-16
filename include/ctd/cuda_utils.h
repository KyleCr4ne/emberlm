#pragma once
// CUDA error-checking macros.

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace ctd {

#define CTD_CUDA_CHECK(expr)                                                       \
  do {                                                                             \
    cudaError_t _err = (expr);                                                     \
    if (_err != cudaSuccess) {                                                     \
      std::ostringstream _oss;                                                     \
      _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " in " << #expr   \
           << " -> " << cudaGetErrorString(_err);                                  \
      throw std::runtime_error(_oss.str());                                        \
    }                                                                              \
  } while (0)

// Use right after a <<<>>> launch.
#define CTD_CUDA_CHECK_KERNEL()                                                    \
  do {                                                                             \
    cudaError_t _err = cudaGetLastError();                                         \
    if (_err != cudaSuccess) {                                                     \
      std::ostringstream _oss;                                                     \
      _oss << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__        \
           << " -> " << cudaGetErrorString(_err);                                  \
      throw std::runtime_error(_oss.str());                                        \
    }                                                                              \
  } while (0)

}  // namespace ctd
