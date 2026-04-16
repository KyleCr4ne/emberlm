#pragma once
// Device descriptor.

#include <cstdint>

namespace ctd {

enum class DeviceType : uint8_t {
  kCPU = 0,
  kCUDA = 1,
};

struct Device {
  DeviceType type = DeviceType::kCPU;
  int index = 0;

  constexpr Device() = default;
  constexpr Device(DeviceType t, int i = 0) : type(t), index(i) {}

  bool is_cpu() const { return type == DeviceType::kCPU; }
  bool is_cuda() const { return type == DeviceType::kCUDA; }

  bool operator==(const Device& o) const { return type == o.type && index == o.index; }
  bool operator!=(const Device& o) const { return !(*this == o); }
};

constexpr Device kCPU{DeviceType::kCPU, 0};
constexpr Device kCUDA0{DeviceType::kCUDA, 0};

}  // namespace ctd
