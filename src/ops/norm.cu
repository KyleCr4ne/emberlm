#include "ctd/ops/norm.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

// One block per row: reduce sum(x^2)/H, compute rsqrt, scale by weight.
template <int kThreads>
__global__ void rms_norm_kernel_f32(const float* __restrict__ x,
                                    const float* __restrict__ w,
                                    float* __restrict__ y,
                                    int H,
                                    float eps) {
  __shared__ float smem[kThreads];
  const int row = blockIdx.x;
  const float* xr = x + row * H;
  float* yr = y + row * H;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < H; i += kThreads) {
    float v = xr[i];
    acc += v * v;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float mean = smem[0] / static_cast<float>(H);
    smem[0] = rsqrtf(mean + eps);
  }
  __syncthreads();
  const float scale = smem[0];

  for (int i = threadIdx.x; i < H; i += kThreads) {
    yr[i] = xr[i] * scale * w[i];
  }
}

// Backward: y_i = x_i * w_i * s, s = rsqrt(mean(x^2) + eps).
// dw_i = sum_rows(g_i * x_i * s)
// dx_i = s * (g_i*w_i - x_i/H * s^2 * sum_j(g_j*w_j*x_j))
// Three passes: recompute s, compute c=sum(g*w*x), write dx and atomicAdd dw.
template <int kThreads>
__global__ void rms_norm_backward_kernel_f32(const float* __restrict__ dy,
                                             const float* __restrict__ x,
                                             const float* __restrict__ w,
                                             float* __restrict__ dx,
                                             float* __restrict__ dw,
                                             int H,
                                             float eps) {
  __shared__ float smem[kThreads];
  const int row = blockIdx.x;
  const float* dyr = dy + row * H;
  const float* xr  = x  + row * H;
  float* dxr       = dx + row * H;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < H; i += kThreads) {
    float v = xr[i];
    acc += v * v;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float mean = smem[0] / static_cast<float>(H);
    smem[0] = rsqrtf(mean + eps);
  }
  __syncthreads();
  const float s = smem[0];

  acc = 0.0f;
  for (int i = threadIdx.x; i < H; i += kThreads) {
    acc += dyr[i] * w[i] * xr[i];
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  const float c = smem[0];
  __syncthreads();

  const float inv_H = 1.0f / static_cast<float>(H);
  for (int i = threadIdx.x; i < H; i += kThreads) {
    float a = dyr[i] * w[i];
    dxr[i] = s * (a - xr[i] * s * s * c * inv_H);
    atomicAdd(&dw[i], dyr[i] * xr[i] * s);
  }
}

}  // namespace

Tensor rms_norm_impl(const Tensor& x, const Tensor& weight, float eps) {
  if (x.dtype() != DType::kFloat32 || weight.dtype() != DType::kFloat32)
    throw std::runtime_error("rms_norm: only float32 is implemented");
  if (!x.device().is_cuda() || !weight.device().is_cuda())
    throw std::runtime_error("rms_norm: only CUDA is implemented");
  if (!x.is_contiguous() || !weight.is_contiguous())
    throw std::runtime_error("rms_norm: non-contiguous inputs not yet supported");
  if (x.dim() < 1) throw std::runtime_error("rms_norm: x must have >= 1 dim");
  const int64_t H = x.shape().back();
  if (weight.dim() != 1 || weight.shape()[0] != H)
    throw std::runtime_error("rms_norm: weight shape mismatch");

  Tensor y = Tensor::empty(x.shape(), x.dtype(), x.device());
  const int64_t rows = x.numel() / H;
  if (rows == 0) return y;

  constexpr int kThreads = 256;
  rms_norm_kernel_f32<kThreads><<<static_cast<unsigned>(rows), kThreads>>>(
      static_cast<const float*>(x.data_ptr()),
      static_cast<const float*>(weight.data_ptr()),
      static_cast<float*>(y.data_ptr()),
      static_cast<int>(H), eps);
  CTD_CUDA_CHECK_KERNEL();
  return y;
}

std::pair<Tensor, Tensor> rms_norm_backward_impl(const Tensor& dy,
                                                 const Tensor& x,
                                                 const Tensor& w,
                                                 float eps) {
  if (dy.shape() != x.shape()) throw std::runtime_error("rms_norm_bwd: dy/x shape mismatch");
  const int64_t H = x.shape().back();
  Tensor dx = Tensor::empty(x.shape(), x.dtype(), x.device());
  Tensor dw = Tensor::zeros({H}, DType::kFloat32, x.device());
  const int64_t rows = x.numel() / H;
  if (rows == 0) return {dx, dw};

  constexpr int kThreads = 256;
  rms_norm_backward_kernel_f32<kThreads><<<static_cast<unsigned>(rows), kThreads>>>(
      static_cast<const float*>(dy.data_ptr()),
      static_cast<const float*>(x.data_ptr()),
      static_cast<const float*>(w.data_ptr()),
      static_cast<float*>(dx.data_ptr()),
      static_cast<float*>(dw.data_ptr()),
      static_cast<int>(H), eps);
  CTD_CUDA_CHECK_KERNEL();
  return {dx, dw};
}

namespace {

struct RmsNormBackward : public autograd::Node {
  Tensor x, w;
  float eps = 1e-5f;
  const char* name() const override { return "RmsNormBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    auto [dx, dw] = rms_norm_backward_impl(go.at(0), x, w, eps);
    return {dx, dw};
  }
};

}  // namespace

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
  Tensor y = rms_norm_impl(x, weight, eps);
  if (autograd::any_requires_grad({x, weight})) {
    auto node = std::make_shared<RmsNormBackward>();
    node->x = x;
    node->w = weight;
    node->eps = eps;
    node->next_edges = autograd::collect_next_edges({x, weight});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
