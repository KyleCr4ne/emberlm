#include "ctd/cublas_utils.h"

#include <mutex>

namespace ctd {

cublasHandle_t cublas_handle() {
  static cublasHandle_t handle = nullptr;
  static std::once_flag flag;
  std::call_once(flag, [] { CTD_CUBLAS_CHECK(cublasCreate(&handle)); });
  return handle;
}

}  // namespace ctd
