#include "ctd/allocator.h"

#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/cuda_utils.h"

namespace ctd {

namespace {

class CudaAllocator final : public Allocator {
 public:
  void* allocate(size_t nbytes) override {
    if (nbytes == 0) return nullptr;
    void* ptr = nullptr;
    CTD_CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  void deallocate(void* ptr) noexcept override {
    if (ptr == nullptr) return;
    // Swallow errors: a failing cudaFree almost always means the context is gone.
    cudaFree(ptr);
  }

  Device device() const override { return kCUDA0; }
};

class CpuAllocator final : public Allocator {
 public:
  void* allocate(size_t nbytes) override {
    if (nbytes == 0) return nullptr;
    void* ptr = std::malloc(nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
    return ptr;
  }
  void deallocate(void* ptr) noexcept override { std::free(ptr); }
  Device device() const override { return kCPU; }
};

// Page-locked host memory — faster cudaMemcpy but cannot be swapped.
class PinnedCpuAllocator final : public Allocator {
 public:
  void* allocate(size_t nbytes) override {
    if (nbytes == 0) return nullptr;
    void* ptr = nullptr;
    CTD_CUDA_CHECK(cudaMallocHost(&ptr, nbytes));
    return ptr;
  }
  void deallocate(void* ptr) noexcept override {
    if (ptr) cudaFreeHost(ptr);
  }
  Device device() const override { return kCPU; }
};

}  // namespace

Allocator& cuda_allocator() {
  static CudaAllocator inst;
  return inst;
}

Allocator& cpu_allocator() {
  static CpuAllocator inst;
  return inst;
}

Allocator& pinned_cpu_allocator() {
  static PinnedCpuAllocator inst;
  return inst;
}

Allocator& allocator_for(const Device& device) {
  switch (device.type) {
    case DeviceType::kCPU: return cpu_allocator();
    case DeviceType::kCUDA: return cuda_allocator();
  }
  throw std::runtime_error("allocator_for: unknown device");
}

}  // namespace ctd
