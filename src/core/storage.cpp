#include "ctd/storage.h"

#include <utility>

namespace ctd {

Storage::Storage(size_t nbytes, Allocator& allocator)
    : allocator_(&allocator), nbytes_(nbytes), device_(allocator.device()) {
  data_ = allocator.allocate(nbytes);
}

Storage::Storage(Storage&& other) noexcept
    : allocator_(other.allocator_),
      data_(other.data_),
      nbytes_(other.nbytes_),
      device_(other.device_) {
  other.allocator_ = nullptr;
  other.data_ = nullptr;
  other.nbytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
  if (this != &other) {
    release_();
    allocator_ = other.allocator_;
    data_ = other.data_;
    nbytes_ = other.nbytes_;
    device_ = other.device_;
    other.allocator_ = nullptr;
    other.data_ = nullptr;
    other.nbytes_ = 0;
  }
  return *this;
}

Storage::~Storage() { release_(); }

void Storage::release_() noexcept {
  if (data_ != nullptr && allocator_ != nullptr) {
    allocator_->deallocate(data_);
  }
  data_ = nullptr;
  nbytes_ = 0;
}

}  // namespace ctd
