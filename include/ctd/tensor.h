#pragma once
// Tensor class.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "ctd/device.h"
#include "ctd/dtype.h"
#include "ctd/storage.h"

namespace ctd {

namespace autograd { struct AutogradMeta; }

// A Tensor is a (storage, layout, dtype) triple.
// Strides are in elements (not bytes). Contiguous row-major: strides[i] = prod(shape[i+1:]).
// Multiple Tensors can share one Storage (views, reshapes, transposes).
class Tensor {
 public:
  Tensor() = default;

  static Tensor empty(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor zeros_pinned(std::vector<int64_t> shape, DType dtype);
  static Tensor from_host(const void* host_data,
                          std::vector<int64_t> shape,
                          DType dtype,
                          Device device);

  void copy_to_host(void* dst) const;

  // Zero-copy view with a new shape. Tensor must be contiguous.
  Tensor reshape(std::vector<int64_t> new_shape) const;

  const std::vector<int64_t>& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }
  int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
  int64_t numel() const;
  size_t nbytes() const { return static_cast<size_t>(numel()) * dtype_size(dtype_); }

  DType dtype() const { return dtype_; }
  const Device& device() const { return device_; }
  bool is_contiguous() const;

  void* data_ptr();
  const void* data_ptr() const;

  const StoragePtr& storage() const { return storage_; }
  int64_t storage_offset() const { return storage_offset_; }

  // ---------------------------------------------------------------------------
  // Autograd interface
  // ---------------------------------------------------------------------------

  Tensor& requires_grad_(bool v = true);
  bool requires_grad() const;

  Tensor grad() const;
  void zero_grad_();

  void backward();
  void backward(const Tensor& grad_output);

  Tensor detach() const;

  const std::shared_ptr<autograd::AutogradMeta>& autograd_meta() const { return autograd_meta_; }
  std::shared_ptr<autograd::AutogradMeta>& mutable_autograd_meta() { return autograd_meta_; }

 private:
  StoragePtr storage_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_ = 0;
  DType dtype_ = DType::kFloat32;
  Device device_{};
  std::shared_ptr<autograd::AutogradMeta> autograd_meta_;
};

// Row-major contiguous strides from shape, in elements.
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);

}  // namespace ctd
