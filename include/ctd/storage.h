#pragma once
// Raw byte buffer on a device.

#include <cstddef>
#include <memory>

#include "ctd/allocator.h"
#include "ctd/device.h"

namespace ctd {

// Owner of one contiguous byte buffer. dtype and shape live in Tensor.
// Multiple Tensors can share one Storage via shared_ptr.
class Storage {
 public:
  Storage(size_t nbytes, Allocator& allocator);

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  Storage(Storage&& other) noexcept;
  Storage& operator=(Storage&& other) noexcept;

  ~Storage();

  void* data() { return data_; }
  const void* data() const { return data_; }
  size_t nbytes() const { return nbytes_; }
  const Device& device() const { return device_; }

 private:
  void release_() noexcept;

  Allocator* allocator_ = nullptr;
  void* data_ = nullptr;
  size_t nbytes_ = 0;
  Device device_{};
};

using StoragePtr = std::shared_ptr<Storage>;

inline StoragePtr make_storage(size_t nbytes, Allocator& alloc) {
  return std::make_shared<Storage>(nbytes, alloc);
}

}  // namespace ctd
