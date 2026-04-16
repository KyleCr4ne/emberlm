#pragma once
// Runtime dtype tag and utilities.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ctd {

enum class DType : uint8_t {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kInt32 = 3,
  kInt64 = 4,
};

constexpr size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::kFloat32: return 4;
    case DType::kFloat16: return 2;
    case DType::kBFloat16: return 2;
    case DType::kInt32: return 4;
    case DType::kInt64: return 8;
  }
  return 0;
}

inline const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::kFloat32: return "float32";
    case DType::kFloat16: return "float16";
    case DType::kBFloat16: return "bfloat16";
    case DType::kInt32: return "int32";
    case DType::kInt64: return "int64";
  }
  return "?";
}

}  // namespace ctd
