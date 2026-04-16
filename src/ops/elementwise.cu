#include "ctd/ops/elementwise.h"

#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

__global__ void add_kernel_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void sub_kernel_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] - b[i];
}

__global__ void mul_kernel_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}

__global__ void scale_kernel_f32(const float* __restrict__ x,
                                 float alpha,
                                 float* __restrict__ out,
                                 int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * x[i];
}

__global__ void silu_kernel_f32(const float* __restrict__ x,
                                float* __restrict__ out,
                                int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    out[i] = v / (1.0f + __expf(-v));
  }
}

// dy * d(silu)/dx; uses expf (not __expf) for numerical stability in gradcheck.
__global__ void silu_backward_kernel_f32(const float* __restrict__ x,
                                         const float* __restrict__ g,
                                         float* __restrict__ out,
                                         int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    float s = 1.0f / (1.0f + expf(-v));
    float dsilu = s + v * s * (1.0f - s);
    out[i] = g[i] * dsilu;
  }
}

void check_binary(const Tensor& a, const Tensor& b, const char* name) {
  if (a.shape() != b.shape()) throw std::runtime_error(std::string(name) + ": shape mismatch");
  if (a.dtype() != b.dtype()) throw std::runtime_error(std::string(name) + ": dtype mismatch");
  if (a.device() != b.device()) throw std::runtime_error(std::string(name) + ": device mismatch");
  if (!a.is_contiguous() || !b.is_contiguous())
    throw std::runtime_error(std::string(name) + ": non-contiguous inputs not yet supported");
  if (a.dtype() != DType::kFloat32)
    throw std::runtime_error(std::string(name) + ": only float32 is implemented");
  if (!a.device().is_cuda())
    throw std::runtime_error(std::string(name) + ": only CUDA is implemented");
}

inline unsigned nblocks_for(int64_t n, int threads) {
  return static_cast<unsigned>((n + threads - 1) / threads);
}

}  // namespace

Tensor add_impl(const Tensor& a, const Tensor& b) {
  check_binary(a, b, "add");
  Tensor out = Tensor::empty(a.shape(), a.dtype(), a.device());
  const int64_t n = a.numel();
  if (n == 0) return out;
  constexpr int kT = 256;
  add_kernel_f32<<<nblocks_for(n, kT), kT>>>(
      static_cast<const float*>(a.data_ptr()),
      static_cast<const float*>(b.data_ptr()),
      static_cast<float*>(out.data_ptr()), n);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

Tensor sub_impl(const Tensor& a, const Tensor& b) {
  check_binary(a, b, "sub");
  Tensor out = Tensor::empty(a.shape(), a.dtype(), a.device());
  const int64_t n = a.numel();
  if (n == 0) return out;
  constexpr int kT = 256;
  sub_kernel_f32<<<nblocks_for(n, kT), kT>>>(
      static_cast<const float*>(a.data_ptr()),
      static_cast<const float*>(b.data_ptr()),
      static_cast<float*>(out.data_ptr()), n);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

Tensor mul_impl(const Tensor& a, const Tensor& b) {
  check_binary(a, b, "mul");
  Tensor out = Tensor::empty(a.shape(), a.dtype(), a.device());
  const int64_t n = a.numel();
  if (n == 0) return out;
  constexpr int kT = 256;
  mul_kernel_f32<<<nblocks_for(n, kT), kT>>>(
      static_cast<const float*>(a.data_ptr()),
      static_cast<const float*>(b.data_ptr()),
      static_cast<float*>(out.data_ptr()), n);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

Tensor scale_impl(const Tensor& x, float alpha) {
  if (x.dtype() != DType::kFloat32 || !x.device().is_cuda() || !x.is_contiguous()) {
    throw std::runtime_error("scale: only fp32 / contiguous / CUDA supported");
  }
  Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
  const int64_t n = x.numel();
  if (n == 0) return out;
  constexpr int kT = 256;
  scale_kernel_f32<<<nblocks_for(n, kT), kT>>>(
      static_cast<const float*>(x.data_ptr()), alpha,
      static_cast<float*>(out.data_ptr()), n);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

Tensor silu_impl(const Tensor& x) {
  if (x.dtype() != DType::kFloat32 || !x.device().is_cuda() || !x.is_contiguous()) {
    throw std::runtime_error("silu: only fp32 / contiguous / CUDA supported");
  }
  Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
  const int64_t n = x.numel();
  if (n == 0) return out;
  constexpr int kT = 256;
  silu_kernel_f32<<<nblocks_for(n, kT), kT>>>(
      static_cast<const float*>(x.data_ptr()),
      static_cast<float*>(out.data_ptr()), n);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

Tensor silu_backward_impl(const Tensor& x, const Tensor& g) {
  check_binary(x, g, "silu_backward");
  Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
  const int64_t n = x.numel();
  if (n == 0) return out;
  constexpr int kT = 256;
  silu_backward_kernel_f32<<<nblocks_for(n, kT), kT>>>(
      static_cast<const float*>(x.data_ptr()),
      static_cast<const float*>(g.data_ptr()),
      static_cast<float*>(out.data_ptr()), n);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

namespace {

struct AddBackward : public autograd::Node {
  const char* name() const override { return "AddBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    const Tensor& dy = go.at(0);
    autograd::NoGradGuard ng;
    return {scale_impl(dy, 1.0f), scale_impl(dy, 1.0f)};
  }
};

struct MulBackward : public autograd::Node {
  Tensor a, b;
  const char* name() const override { return "MulBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    const Tensor& dy = go.at(0);
    autograd::NoGradGuard ng;
    return {mul_impl(dy, b), mul_impl(dy, a)};
  }
};

struct SiluBackward : public autograd::Node {
  Tensor x;
  const char* name() const override { return "SiluBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    return {silu_backward_impl(x, go.at(0))};
  }
};

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor y = add_impl(a, b);
  if (autograd::any_requires_grad({a, b})) {
    auto node = std::make_shared<AddBackward>();
    node->next_edges = autograd::collect_next_edges({a, b});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor y = mul_impl(a, b);
  if (autograd::any_requires_grad({a, b})) {
    auto node = std::make_shared<MulBackward>();
    node->a = a;
    node->b = b;
    node->next_edges = autograd::collect_next_edges({a, b});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

Tensor silu(const Tensor& x) {
  Tensor y = silu_impl(x);
  if (autograd::any_requires_grad({x})) {
    auto node = std::make_shared<SiluBackward>();
    node->x = x;
    node->next_edges = autograd::collect_next_edges({x});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
