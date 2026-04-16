#pragma once
// Memory allocator interface.

#include <cstddef>

#include "ctd/device.h"

namespace ctd {

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* allocate(size_t nbytes) = 0;
  virtual void deallocate(void* ptr) noexcept = 0;
  virtual Device device() const = 0;
};

Allocator& cuda_allocator();
Allocator& cpu_allocator();
Allocator& pinned_cpu_allocator();
Allocator& allocator_for(const Device& device);

}  // namespace ctd
