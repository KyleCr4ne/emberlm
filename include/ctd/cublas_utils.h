#pragma once
// cuBLAS handle and error-checking macros.

#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>

namespace ctd {

inline const char* cublas_status_string(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "LICENSE_ERROR";
    default: return "UNKNOWN";
  }
}

#define CTD_CUBLAS_CHECK(expr)                                                 \
  do {                                                                         \
    cublasStatus_t _s = (expr);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      std::ostringstream _oss;                                                 \
      _oss << "cuBLAS error at " << __FILE__ << ":" << __LINE__                \
           << " in " << #expr << " -> " << cublas_status_string(_s);           \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

// Lazily-initialized process-wide cuBLAS handle.
cublasHandle_t cublas_handle();

}  // namespace ctd
