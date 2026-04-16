#include "ctd/ops/softmax.h"

#include <cfloat>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

// One block per row: row max then sum-exp normalize.
template <int kThreads>
__global__ void softmax_kernel_f32(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int N) {
  __shared__ float smem[kThreads];
  const int row = blockIdx.x;
  const float* xr = x + row * N;
  float* yr = y + row * N;

  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < N; i += kThreads) m = fmaxf(m, xr[i]);
  smem[threadIdx.x] = m;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  const float row_max = smem[0];
  __syncthreads();

  float acc = 0.0f;
  for (int i = threadIdx.x; i < N; i += kThreads) {
    float e = __expf(xr[i] - row_max);
    yr[i] = e;
    acc += e;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  const float inv_sum = 1.0f / smem[0];

  for (int i = threadIdx.x; i < N; i += kThreads) yr[i] *= inv_sum;
}

// Backward: dx_i = y_i * (g_i - sum_j(g_j * y_j)).
template <int kThreads>
__global__ void softmax_backward_kernel_f32(const float* __restrict__ y,
                                            const float* __restrict__ g,
                                            float* __restrict__ dx,
                                            int N) {
  __shared__ float smem[kThreads];
  const int row = blockIdx.x;
  const float* yr = y  + row * N;
  const float* gr = g  + row * N;
  float* dxr      = dx + row * N;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < N; i += kThreads) acc += gr[i] * yr[i];
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  const float sumgy = smem[0];
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += kThreads) {
    dxr[i] = yr[i] * (gr[i] - sumgy);
  }
}

}  // namespace

Tensor softmax_impl(const Tensor& x) {
  if (x.dtype() != DType::kFloat32 || !x.device().is_cuda() || !x.is_contiguous())
    throw std::runtime_error("softmax: only fp32 / contiguous / CUDA supported");
  if (x.dim() < 1) throw std::runtime_error("softmax: need >= 1 dim");
  Tensor y = Tensor::empty(x.shape(), x.dtype(), x.device());
  const int N = static_cast<int>(x.shape().back());
  const int64_t rows = x.numel() / N;
  if (rows == 0) return y;
  constexpr int kThreads = 256;
  softmax_kernel_f32<kThreads><<<static_cast<unsigned>(rows), kThreads>>>(
      static_cast<const float*>(x.data_ptr()),
      static_cast<float*>(y.data_ptr()), N);
  CTD_CUDA_CHECK_KERNEL();
  return y;
}

Tensor softmax_backward_impl(const Tensor& y, const Tensor& g) {
  if (y.shape() != g.shape()) throw std::runtime_error("softmax_bwd: shape mismatch");
  Tensor dx = Tensor::empty(y.shape(), y.dtype(), y.device());
  const int N = static_cast<int>(y.shape().back());
  const int64_t rows = y.numel() / N;
  if (rows == 0) return dx;
  constexpr int kThreads = 256;
  softmax_backward_kernel_f32<kThreads><<<static_cast<unsigned>(rows), kThreads>>>(
      static_cast<const float*>(y.data_ptr()),
      static_cast<const float*>(g.data_ptr()),
      static_cast<float*>(dx.data_ptr()), N);
  CTD_CUDA_CHECK_KERNEL();
  return dx;
}

namespace {
struct SoftmaxBackward : public autograd::Node {
  Tensor y;
  const char* name() const override { return "SoftmaxBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    return {softmax_backward_impl(y, go.at(0))};
  }
};
}  // namespace

Tensor softmax(const Tensor& x) {
  Tensor y = softmax_impl(x);
  if (autograd::any_requires_grad({x})) {
    auto node = std::make_shared<SoftmaxBackward>();
    node->y = y;
    node->next_edges = autograd::collect_next_edges({x});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
