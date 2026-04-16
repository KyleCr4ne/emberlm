#include "ctd/tensor.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/allocator.h"
#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd {

namespace {

class ReshapeBackward : public autograd::Node {
 public:
  std::vector<int64_t> input_shape;
  std::vector<Tensor> backward(const std::vector<Tensor>& grads_out) override {
    if (grads_out.size() != 1) throw std::runtime_error("ReshapeBackward: expected 1 grad");
    return {grads_out[0].reshape(input_shape)};
  }
  const char* name() const override { return "ReshapeBackward"; }
};

}  // namespace

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> s(shape.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

int64_t Tensor::numel() const {
  if (shape_.empty()) return 1;
  int64_t n = 1;
  for (auto d : shape_) n *= d;
  return n;
}

bool Tensor::is_contiguous() const {
  return strides_ == contiguous_strides(shape_);
}

void* Tensor::data_ptr() {
  if (storage_ == nullptr) return nullptr;
  auto* base = static_cast<std::byte*>(storage_->data());
  return base + static_cast<size_t>(storage_offset_) * dtype_size(dtype_);
}

const void* Tensor::data_ptr() const {
  if (storage_ == nullptr) return nullptr;
  const auto* base = static_cast<const std::byte*>(storage_->data());
  return base + static_cast<size_t>(storage_offset_) * dtype_size(dtype_);
}

Tensor Tensor::empty(std::vector<int64_t> shape, DType dtype, Device device) {
  Tensor t;
  t.shape_ = std::move(shape);
  t.strides_ = contiguous_strides(t.shape_);
  t.dtype_ = dtype;
  t.device_ = device;
  const size_t nbytes = static_cast<size_t>(t.numel()) * dtype_size(dtype);
  t.storage_ = make_storage(nbytes, allocator_for(device));
  return t;
}

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype, Device device) {
  Tensor t = empty(std::move(shape), dtype, device);
  if (t.nbytes() == 0) return t;
  if (device.is_cuda()) {
    CTD_CUDA_CHECK(cudaMemset(t.data_ptr(), 0, t.nbytes()));
  } else {
    std::memset(t.data_ptr(), 0, t.nbytes());
  }
  return t;
}

Tensor Tensor::zeros_pinned(std::vector<int64_t> shape, DType dtype) {
  Tensor t;
  t.shape_ = std::move(shape);
  t.strides_ = contiguous_strides(t.shape_);
  t.dtype_ = dtype;
  t.device_ = kCPU;
  const size_t nbytes = static_cast<size_t>(t.numel()) * dtype_size(dtype);
  t.storage_ = make_storage(nbytes, pinned_cpu_allocator());
  if (nbytes > 0) std::memset(t.data_ptr(), 0, nbytes);
  return t;
}

Tensor Tensor::from_host(const void* host_data,
                         std::vector<int64_t> shape,
                         DType dtype,
                         Device device) {
  Tensor t = empty(std::move(shape), dtype, device);
  if (t.nbytes() == 0) return t;
  if (device.is_cuda()) {
    CTD_CUDA_CHECK(cudaMemcpy(t.data_ptr(), host_data, t.nbytes(), cudaMemcpyHostToDevice));
  } else {
    std::memcpy(t.data_ptr(), host_data, t.nbytes());
  }
  return t;
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
  if (!is_contiguous()) throw std::runtime_error("reshape: input must be contiguous");
  int64_t new_numel = 1;
  for (auto d : new_shape) new_numel *= d;
  if (new_numel != numel()) throw std::runtime_error("reshape: numel mismatch");
  Tensor out;
  out.storage_ = storage_;
  out.storage_offset_ = storage_offset_;
  out.dtype_ = dtype_;
  out.device_ = device_;
  out.shape_ = new_shape;
  out.strides_ = contiguous_strides(out.shape_);

  if (autograd::any_requires_grad({*this})) {
    auto node = std::make_shared<ReshapeBackward>();
    node->input_shape = shape_;
    node->next_edges = autograd::collect_next_edges({*this});
    autograd::set_history(out, std::move(node));
  }
  return out;
}

void Tensor::copy_to_host(void* dst) const {
  if (nbytes() == 0) return;
  if (!is_contiguous()) {
    throw std::runtime_error("Tensor::copy_to_host: non-contiguous tensors not yet supported");
  }
  if (device_.is_cuda()) {
    CTD_CUDA_CHECK(cudaMemcpy(dst, data_ptr(), nbytes(), cudaMemcpyDeviceToHost));
  } else {
    std::memcpy(dst, data_ptr(), nbytes());
  }
}

}  // namespace ctd
