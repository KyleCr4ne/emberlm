#include "ctd/ops/repeat_kv.h"

#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

// One block per (bt, h_out); copies D elements from input head h_out/group.
__global__ void repeat_kv_kernel_f32(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int64_t BT, int H_kv, int group, int D) {
  int bt = blockIdx.y;
  int h_out = blockIdx.x;
  int h_in = h_out / group;

  const float* src = x + ((int64_t)bt * H_kv + h_in) * D;
  float* dst = y + ((int64_t)bt * H_kv * group + h_out) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

// Backward: sum the `group` output heads back into a single input head.
__global__ void repeat_kv_sum_kernel_f32(const float* __restrict__ y,
                                         float* __restrict__ dx,
                                         int64_t BT, int H_kv, int group, int D) {
  int bt = blockIdx.y;
  int h_in = blockIdx.x;
  const int H_out = H_kv * group;

  const float* src_base = y + ((int64_t)bt * H_out + h_in * group) * D;
  float* dst = dx + ((int64_t)bt * H_kv + h_in) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < group; ++k) acc += src_base[k * D + d];
    dst[d] = acc;
  }
}

__global__ void mul_scalar_kernel_f32(const float* __restrict__ x,
                                      float c,
                                      float* __restrict__ y,
                                      int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * c;
}

}  // namespace

Tensor repeat_kv_heads_impl(const Tensor& x, int group) {
  if (x.dtype() != DType::kFloat32 || !x.device().is_cuda() || !x.is_contiguous()) {
    throw std::runtime_error("repeat_kv: only fp32 / contiguous / CUDA supported");
  }
  if (x.dim() != 4) throw std::runtime_error("repeat_kv: expected [B, T, H_kv, D]");
  if (group <= 0) throw std::runtime_error("repeat_kv: group must be > 0");

  const int64_t B = x.shape()[0];
  const int64_t T = x.shape()[1];
  const int H_kv = static_cast<int>(x.shape()[2]);
  const int D = static_cast<int>(x.shape()[3]);
  const int H_out = H_kv * group;

  Tensor y = Tensor::empty({B, T, H_out, D}, DType::kFloat32, x.device());
  const int64_t BT = B * T;
  if (BT == 0) return y;

  dim3 grid(static_cast<unsigned>(H_out), static_cast<unsigned>(BT));
  const int kThreads = 64;
  repeat_kv_kernel_f32<<<grid, kThreads>>>(
      static_cast<const float*>(x.data_ptr()),
      static_cast<float*>(y.data_ptr()),
      BT, H_kv, group, D);
  CTD_CUDA_CHECK_KERNEL();
  return y;
}

Tensor repeat_kv_sum_impl(const Tensor& y, int group) {
  if (y.dtype() != DType::kFloat32 || !y.device().is_cuda() || !y.is_contiguous())
    throw std::runtime_error("repeat_kv_sum: fp32 / contiguous / CUDA only");
  if (y.dim() != 4) throw std::runtime_error("repeat_kv_sum: expected 4D");
  const int64_t B = y.shape()[0];
  const int64_t T = y.shape()[1];
  const int H_out = static_cast<int>(y.shape()[2]);
  const int D = static_cast<int>(y.shape()[3]);
  if (H_out % group != 0) throw std::runtime_error("repeat_kv_sum: H_out not divisible by group");
  const int H_kv = H_out / group;

  Tensor dx = Tensor::empty({B, T, H_kv, D}, DType::kFloat32, y.device());
  const int64_t BT = B * T;
  if (BT == 0) return dx;

  dim3 grid(static_cast<unsigned>(H_kv), static_cast<unsigned>(BT));
  const int kThreads = 64;
  repeat_kv_sum_kernel_f32<<<grid, kThreads>>>(
      static_cast<const float*>(y.data_ptr()),
      static_cast<float*>(dx.data_ptr()),
      BT, H_kv, group, D);
  CTD_CUDA_CHECK_KERNEL();
  return dx;
}

Tensor mul_scalar_impl(const Tensor& x, float c) {
  if (x.dtype() != DType::kFloat32 || !x.device().is_cuda() || !x.is_contiguous()) {
    throw std::runtime_error("mul_scalar: only fp32 / contiguous / CUDA supported");
  }
  Tensor y = Tensor::empty(x.shape(), x.dtype(), x.device());
  const int64_t n = x.numel();
  if (n == 0) return y;
  constexpr int kThreads = 256;
  const int64_t nblocks = (n + kThreads - 1) / kThreads;
  mul_scalar_kernel_f32<<<static_cast<unsigned>(nblocks), kThreads>>>(
      static_cast<const float*>(x.data_ptr()), c,
      static_cast<float*>(y.data_ptr()), n);
  CTD_CUDA_CHECK_KERNEL();
  return y;
}

namespace {

struct RepeatKvBackward : public autograd::Node {
  int group = 1;
  const char* name() const override { return "RepeatKvBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    return {repeat_kv_sum_impl(go.at(0), group)};
  }
};

struct MulScalarBackward : public autograd::Node {
  float c = 1.0f;
  const char* name() const override { return "MulScalarBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    return {mul_scalar_impl(go.at(0), c)};
  }
};

}  // namespace

Tensor repeat_kv_heads(const Tensor& x, int group) {
  Tensor y = repeat_kv_heads_impl(x, group);
  if (autograd::any_requires_grad({x})) {
    auto node = std::make_shared<RepeatKvBackward>();
    node->group = group;
    node->next_edges = autograd::collect_next_edges({x});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

Tensor mul_scalar(const Tensor& x, float c) {
  Tensor y = mul_scalar_impl(x, c);
  if (autograd::any_requires_grad({x})) {
    auto node = std::make_shared<MulScalarBackward>();
    node->c = c;
    node->next_edges = autograd::collect_next_edges({x});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
